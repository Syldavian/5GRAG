#!/usr/bin/env python3
"""
Test Case Gap Discovery System
Automated identification of test case gaps between O-RAN specifications and 3GPP references
"""

import re
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from controller import Controller
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from settings import config

class TestCaseGapDiscovery:
    """Discover potential test case gaps between O‑RAN specs and 3GPP refs."""

    # ---------------------------------------------------------------------
    # 1. INITIALISATION & PROMPTS
    # ---------------------------------------------------------------------
    def __init__(self) -> None:
        self.controller = Controller()
        self.llm_parser = ChatOpenAI(api_key=config["API_KEY"], model="gpt-4.1-mini")
        # Full model for reasoning and analysis
        self.llm_full = ChatOpenAI(api_key=config["API_KEY"], model=config["MODEL_NAME"])

        # ── 1A  Sentence‑level entity parser (unchanged) ──────────────────
        self.parser_prompt = ChatPromptTemplate.from_template(
            """
            Given the following sentence from an O‑RAN specification:
            "{sentence}"

            Extract the following information in JSON format:
            - "acting_component": The component performing the action (e.g., "gNB‑DU", "gNB‑CU‑CP")
            - "trigger_message": The message or event that initiates the action (e.g., "TRACE START message")
            - "primary_action": The main verb or action phrase (e.g., "initiate trace session")
            - "key_parameters_mentioned_in_sentence": A list of important parameters or conditions explicitly mentioned
            - "referenced_3gpp_document": The full 3GPP document number if found

            Return only valid JSON without any additional text or explanation.
            """
        )

        # ── 1B  NEW PROMPT: interpret what the 3GPP reference implies ─────
        self.reference_analysis_prompt = ChatPromptTemplate.from_template(
            """
            You are a 5G/NR specification expert. Your task is to interpret what the
            following O‑RAN requirement means in the light of the cited 3GPP specification.

            **O‑RAN requirement sentence**
            "{original_oran_sentence}"

            **Relevant excerpts from 3GPP document ({referenced_3gpp_document})**
            {combined_3gpp_context}

            From the 3GPP excerpts, produce a comprehensive list of **specific, testable
            items** (conditions, parameter variations, procedural steps, abnormal cases)
            that the "{acting_component}" must fulfil when performing "{primary_action}", 
            based only on what is explicitly supported by the provided 3GPP excerpts.

            **Output (strict JSON):**
            {{
              "analyzed_requirement": "{original_oran_sentence}",
              "3gpp_document": "{referenced_3gpp_document}",
              "detailed_3gpp_testable_items": [
                {{
                  "item_description": "Specific condition / parameter / procedure",
                  "justification": "Short explanation linking back to 3GPP text"
                }}
              ]
            }}

            Return **ONLY** valid JSON – no extra commentary.
            """
        )

        # ── 1C  NEW PROMPT: evaluate coverage & identify gaps ─────────────
        self.gap_analysis_prompt = ChatPromptTemplate.from_template(
            """
            You are an O‑RAN test‑case analyst. Using the derived list of testable items
            and the current O‑RAN test‑case excerpts, determine coverage and gaps.

            **O‑RAN requirement sentence**: "{original_oran_sentence}"

            **Testable items (JSON)**:
            {testable_items_json}

            **O‑RAN test‑case excerpts**:
            {combined_oran_test_context}

            For **each** testable item decide if it is already covered by the O‑RAN
            test‑cases. Mark as "Covered", "Partially Covered", or "Not Covered" and
            give a short justification.

            Then list every item that is not fully covered as a **potential test case
            gap** and suggest a concise test‑scenario concept.

            **Output (strict JSON):**
            {{
              "detailed_3gpp_testable_items": [
                {{
                  "item_description": "…",
                  "oran_test_coverage": "Covered/Partially Covered/Not Covered",
                  "coverage_justification": "…"
                }}
              ],
              "potential_test_case_gaps": [
                {{
                  "gap_description": "…",
                  "suggested_test_scenario_concept": "…"
                }}
              ]
            }}

            Return **ONLY** valid JSON.
            """
        )

    def extract_3gpp_reference(self, sentence: str) -> Optional[str]:
        """
        Step 1: Extract 3GPP document reference from sentence
        """
        print("\n=== STEP 1: Input Processor & Parser ===")
        print(f"Analyzing sentence: {sentence}")
        
        # Common patterns for 3GPP references
        patterns = [
            r'TS\s+(\d+\.\d+)',  # TS 32.422
            r'3GPP\s+TS\s+(\d+\.\d+)',  # 3GPP TS 32.422
            r'\[(\d+)\].*?TS\s+(\d+\.\d+)',  # [29] ... TS 32.422
            r'TS\s+(\d+\.\d+)\s+\[(\d+)\]'   # TS 32.422 [29]
        ]
        
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                if len(match.groups()) == 1:
                    doc_ref = f"TS {match.group(1)}"
                else:
                    doc_ref = f"TS {match.group(1) if match.group(1).replace('.', '').isdigit() else match.group(2)}"
                print(f"✓ Found 3GPP reference: {doc_ref}")
                return doc_ref
        
        print("✗ No 3GPP reference found")
        return None

    def parse_oran_sentence(self, sentence: str) -> Dict:
        """
        Step 1: Use LLM to parse O-RAN sentence and extract entities
        """
        print("\n--- Semantic Parsing with LLM ---")
        
        try:
            chain = self.parser_prompt | self.llm_parser
            response = chain.invoke({"sentence": sentence})
            
            # Clean the response and parse JSON
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            
            parsed_data = json.loads(content)
            print(f"✓ Parsed entities: {json.dumps(parsed_data, indent=2)}")
            return parsed_data
            
        except Exception as e:
            print(f"✗ Error parsing sentence: {e}")
            return {
                "acting_component": "Unknown",
                "trigger_message": "Unknown",
                "primary_action": "Unknown",
                "key_parameters_mentioned_in_sentence": [],
                "referenced_3gpp_document": ""
            }

    def query_3gpp_context(self, parsed_data: Dict, doc_ref: str, original_sentence: str) -> List[str]:
        """
        Step 2A: Query 3GPP documents using RAG with reference extraction
        """
        print(f"\n=== STEP 2A: Querying 3GPP Document ({doc_ref}) ===")
        
        # Import reference extractor
        from ReferenceExtractor import ReferenceExtractor
        from langchain_core.documents import Document
        import os
        
        RExt = ReferenceExtractor()
        
        # Extract document IDs and clause numbers from the original sentence
        print("--- Reference Extraction ---")
        doc_ids = RExt.extractDocIdsFromStrList([original_sentence])
        
        # Create a document object for clause extraction
        prompt_doc = Document(page_content=original_sentence)
        clause_nums = RExt.extractClauseNumbersOfSrc(RExt.runREWithDocList([prompt_doc]))
        
        print(f"📘 Detected doc IDs: {doc_ids}")
        print(f"📑 Detected clause numbers: {clause_nums}")
        
        # Get all available documents in DOC_DIR for mapping
        all_docs = []
        try:
            all_docs = [f for f in os.listdir(config["DOC_DIR"]) if f.endswith('.docx')]
        except Exception as e:
            print(f"Warning: Could not list documents in {config.get('DOC_DIR', 'DOC_DIR')}: {e}")
        
        # Map doc IDs to actual filenames
        def doc_id_to_filename(doc_id: str, available_docs: List[str]) -> Optional[str]:
            """Map a document ID like 'TS 32.422' to actual filename"""
            # Clean the doc_id - remove 'TS ' prefix and get numbers
            clean_id = doc_id.replace('TS ', '').replace('TR ', '')
            
            for doc in available_docs:
                # Check if document name contains the ID
                if clean_id.replace('.', '') in doc.replace('-', '').replace('.', ''):
                    return os.path.join(config["DOC_DIR"], doc)
                # Also check with dots
                if clean_id in doc:
                    return os.path.join(config["DOC_DIR"], doc)
            return None
        
        extra_filtered_docs = [doc_id_to_filename(doc_id, all_docs) for doc_id in doc_ids]
        extra_filtered_docs = [doc for doc in extra_filtered_docs if doc]  # Remove None
        
        print(f"📁 Mapped to files: {[os.path.basename(doc) for doc in extra_filtered_docs] if extra_filtered_docs else 'None'}")
        
        # Use context database for 3GPP specs
        self.controller.switchDatabase("context")
        
        # Construct multiple queries for comprehensive retrieval
        base_queries = [
            f"{doc_ref} {parsed_data.get('trigger_message', '')} {parsed_data.get('acting_component', '')}",
            f"{doc_ref} {parsed_data.get('primary_action', '')}",
            f"{doc_ref} {' '.join(parsed_data.get('key_parameters_mentioned_in_sentence', []))}",
            f"{doc_ref} abnormal conditions",
            f"{doc_ref} error cases"
        ]
        
        # Add clause-specific queries if clause numbers were detected
        if clause_nums:
            for clause in clause_nums:
                base_queries.append(f"{doc_ref} clause {clause}")
                base_queries.append(f"section {clause}")
        
        all_contexts = []
        
        # Query with document filtering if we found specific documents
        if extra_filtered_docs:
            print(f"--- Querying with document scope filtering ---")
            selected_docs = [os.path.basename(doc) for doc in extra_filtered_docs]
            
            for query in base_queries:
                query = query.strip()
                if not query or query == doc_ref:
                    continue
                    
                print(f"Querying (filtered): {query}")
                try:
                    # Use the controller's runController method with document filtering
                    response, orig_docs, additional_docs = self.controller.runController(query, [], selected_docs)
                    
                    # Extract contexts from retrieved documents
                    contexts = []
                    for doc in orig_docs + additional_docs:
                        # Check if document is relevant to our reference
                        source = doc.metadata.get('source', '')
                        if any(doc_id.replace('TS ', '').replace('.', '') in source.replace('-', '').replace('.', '') for doc_id in doc_ids):
                            contexts.append(f"Source: {source}\nSection: {doc.metadata.get('section', 'Unknown')}\n{doc.page_content}")
                    
                    all_contexts.extend(contexts)
                    print(f"  Retrieved {len(contexts)} relevant chunks from filtered docs")
                    
                except Exception as e:
                    print(f"  Error querying with filtering: {e}")
        
        else:
            print(f"--- Querying without document filtering (fallback) ---")
            # Fallback to original approach if no specific documents found
            for query in base_queries:
                query = query.strip()
                if not query or query == doc_ref:
                    continue
                    
                print(f"Querying (unfiltered): {query}")
                try:
                    response, orig_docs, additional_docs = self.controller.getResponseWithRetrieval(query, [])
                    
                    # Extract contexts from retrieved documents
                    contexts = []
                    for doc in orig_docs + additional_docs:
                        # Check if document is relevant to our reference
                        source = doc.metadata.get('source', '')
                        doc_ref_clean = doc_ref.replace("TS ", "").replace(".", "")
                        if doc_ref_clean in source.replace('-', '').replace('.', ''):
                            contexts.append(f"Source: {source}\nSection: {doc.metadata.get('section', 'Unknown')}\n{doc.page_content}")
                    
                    all_contexts.extend(contexts)
                    print(f"  Retrieved {len(contexts)} relevant chunks")
                    
                except Exception as e:
                    print(f"  Error querying: {e}")
        
        # Add clause-specific filtering if clause numbers were detected
        if clause_nums and all_contexts:
            print(f"--- Filtering by clause numbers: {clause_nums} ---")
            clause_filtered_contexts = []
            for ctx in all_contexts:
                # Check if any of the detected clause numbers appear in the context
                for clause in clause_nums:
                    if f"clause {clause}" in ctx.lower() or f"section {clause}" in ctx.lower() or f"{clause}." in ctx:
                        clause_filtered_contexts.append(ctx)
                        break
            
            if clause_filtered_contexts:
                print(f"  Filtered to {len(clause_filtered_contexts)} clause-specific chunks")
                all_contexts = clause_filtered_contexts
        
        # Remove duplicates while preserving order
        unique_contexts = []
        seen = set()
        for ctx in all_contexts:
            if ctx not in seen:
                unique_contexts.append(ctx)
                seen.add(ctx)
        
        print(f"✓ Total unique 3GPP contexts retrieved: {len(unique_contexts)}")
        return unique_contexts

    def query_oran_test_context(self, parsed_data: Dict) -> List[str]:
        """
        Step 2B: Query O-RAN test specifications using RAG
        """
        print(f"\n=== STEP 2B: Querying O-RAN Test Specifications ===")
        
        # Construct queries for O-RAN test cases
        queries = [
            f"O-RAN test {parsed_data.get('acting_component', '')} {parsed_data.get('trigger_message', '')}",
            f"test case {parsed_data.get('primary_action', '')}",
            f"O-RAN test {' '.join(parsed_data.get('key_parameters_mentioned_in_sentence', []))}",
            f"test scenario {parsed_data.get('acting_component', '')}"
        ]
        
        # Use testcase database for O-RAN test specs
        self.controller.switchDatabase("testcase")
        
        all_contexts = []
        for query in queries:
            query = query.strip()
            if not query:
                continue
                
            print(f"Querying: {query}")
            try:
                response, orig_docs, additional_docs = self.controller.getResponseWithRetrieval(query, [])
                
                # Extract contexts from retrieved documents
                contexts = []
                for doc in orig_docs + additional_docs:
                    contexts.append(f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}")
                
                all_contexts.extend(contexts)
                print(f"  Retrieved {len(contexts)} test case chunks")
                
            except Exception as e:
                print(f"  Error querying: {e}")
        
        # Remove duplicates
        unique_contexts = []
        seen = set()
        for ctx in all_contexts:
            if ctx not in seen:
                unique_contexts.append(ctx)
                seen.add(ctx)
        
        print(f"✓ Total unique O-RAN test contexts retrieved: {len(unique_contexts)}")
        return unique_contexts

    def combine_contexts(self, contexts_3gpp: List[str], contexts_oran: List[str]) -> tuple:
        """
        Step 3: Combine retrieved contexts
        """
        print(f"\n=== STEP 3: Context Combiner ===")
        
        combined_3gpp = "\n\n---\n\n".join(contexts_3gpp) if contexts_3gpp else "No relevant 3GPP context found."
        combined_oran = "\n\n---\n\n".join(contexts_oran) if contexts_oran else "No relevant O-RAN test cases found."
        
        print(f"✓ Combined 3GPP context: {len(combined_3gpp)} characters")
        print(f"✓ Combined O-RAN test context: {len(combined_oran)} characters")
        
        return combined_3gpp, combined_oran
    
    # ---------------------------------------------------------------------
    # 3. ANALYSIS ENGINE – now two‑phase
    # ---------------------------------------------------------------------
    def _interpret_reference(self, original_sentence: str, parsed_data: Dict, combined_3gpp_context: str) -> Dict:
        """Phase 1 – what does the 3GPP reference actually require?"""
        chain = self.reference_analysis_prompt | self.llm_full
        raw = chain.invoke({
            "original_oran_sentence": original_sentence,
            "referenced_3gpp_document": parsed_data.get("referenced_3gpp_document", "Unknown"),
            "combined_3gpp_context": combined_3gpp_context,
            "primary_action": parsed_data.get("primary_action", ""),
            "acting_component": parsed_data.get("acting_component", "")
        }).content.strip()
        raw = raw[7:-3].strip() if raw.startswith("```json") else raw
        return json.loads(raw)

    def _analyze_gaps(self, original_sentence: str, testable_items: Dict,
                       combined_oran_test_context: str) -> Dict:
        """Phase 2 – compare with existing O‑RAN test cases and find gaps."""
        chain = self.gap_analysis_prompt | self.llm_full
        raw = chain.invoke({
            "original_oran_sentence": original_sentence,
            "testable_items_json": json.dumps(testable_items, indent=2),
            "combined_oran_test_context": combined_oran_test_context
        }).content.strip()
        raw = raw[7:-3].strip() if raw.startswith("```json") else raw
        return json.loads(raw)

    def run_gap_discovery(self, oran_sentence: str) -> Dict:
        # … (Step 1 parsing, Step 2 context retrieval – unchanged) …

        doc_ref = self.extract_3gpp_reference(oran_sentence)
        if not doc_ref:
            return {"error": "No 3GPP reference found"}
        parsed = self.parse_oran_sentence(oran_sentence)
        parsed.setdefault("referenced_3gpp_document", doc_ref)

        ctx_3gpp = self.query_3gpp_context(parsed, doc_ref, oran_sentence)
        ctx_oran = self.query_oran_test_context(parsed)
        combined_3gpp, combined_oran = self.combine_contexts(ctx_3gpp, ctx_oran)

        # ── Phase 1 – reference interpretation ───────────────────────────
        reference_info = self._interpret_reference(oran_sentence, parsed, combined_3gpp)

        # ── Phase 2 – gap analysis against test cases ─────────────────────
        gap_analysis = self._analyze_gaps(oran_sentence, reference_info, combined_oran)

        # Merge for convenience
        return {
            "original_sentence": oran_sentence,
            "parsed_entities": parsed,
            "reference_interpretation": reference_info,
            "gap_analysis": gap_analysis
        }

    # ------------------------------------------------------------------
    # 5. RESULTS PRESENTATION – now also persists to timestamped file
    # ------------------------------------------------------------------
    def display_results(self, result: Dict) -> None:
        if "error" in result:
            print("❌", result["error"])
            return

        # Pretty‑print to console (truncated for brevity)
        print("\n📝 Requirement:", result["original_sentence"])
        print("\n📑 Parsed entities:")
        for k, v in result["parsed_entities"].items():
            print(f"   {k}: {v}")

        print("\n📊 3GPP‑derived testable items:")
        for itm in result["reference_interpretation"].get("detailed_3gpp_testable_items", []):
            print(" •", itm["item_description"], "–", itm.get("justification", ""))

        print("\n🚨 Potential gaps:")
        gaps = result["gap_analysis"].get("potential_test_case_gaps", [])
        if not gaps:
            print("   ✅ No gaps – all items covered.")
        else:
            for g in gaps:
                print(" •", g["gap_description"])

        # ── NEW: persist to file ─────────────────────────────────────────
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_fname = f"gap_analysis_{timestamp}.json"
        with open(out_fname, "w", encoding="utf‑8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        print(f"\n📁 Detailed JSON report written to {out_fname}\n")


def main():
    """
    Main CLI interface for the test case gap discovery system
    """
    print("🔍 Test Case Gap Discovery System")
    print("Identifies potential gaps between O-RAN specs and 3GPP references")
    print("-" * 60)
    
    gap_discovery = TestCaseGapDiscovery()
    
    while True:
        print("\nEnter an O-RAN specification sentence with a 3GPP reference")
        print("(or 'quit' to exit, 'example' for a sample sentence):")
        
        user_input = input("\n> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'example':
            user_input = ("Upon reception of the TRACE START message, the gNB-DU shall "
                         "initiate the requested trace session for the requested UE, "
                         "as described in TS 32.422 [29].")
            print(f"Using example: {user_input}")
        
        if not user_input:
            continue
        
        try:
            # Run the gap discovery process
            result = gap_discovery.run_gap_discovery(user_input)
            
            # Display formatted results
            gap_discovery.display_results(result)
            
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

def run_batch(predefined_sentences):
    """
    Main CLI interface for the test case gap discovery system
    """
    print("🔍 Test Case Gap Discovery System")
    print("Identifies potential gaps between O-RAN specs and 3GPP references")
    print("-" * 60)

    gap_discovery = TestCaseGapDiscovery()

    if predefined_sentences:
        for idx, sentence in enumerate(predefined_sentences, start=1):
            print(f"\n🔎 Processing sentence {idx}/{len(predefined_sentences)}:")
            print(f"» {sentence}")
            try:
                result = gap_discovery.run_gap_discovery(sentence)
                gap_discovery.display_results(result)
            except Exception as e:
                print(f"❌ Error processing sentence {idx}: {e}")
        print("\n✅ Finished processing predefined sentences.")
        return  # Exit after processing predefined list

    # 🧑‍💻 Fallback to interactive mode
    while True:
        print("\nEnter an O-RAN specification sentence with a 3GPP reference")
        print("(or 'quit' to exit, 'example' for a sample sentence):")

        user_input = input("\n> ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if user_input.lower() == 'example':
            user_input = ("Upon reception of the TRACE START message, the gNB-DU shall "
                          "initiate the requested trace session for the requested UE, "
                          "as described in TS 32.422 [29].")
            print(f"Using example: {user_input}")

        if not user_input:
            continue

        try:
            result = gap_discovery.run_gap_discovery(user_input)
            gap_discovery.display_results(result)
        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue

if __name__ == "__main__":
    #main()
    f1_sentences = [
        "If the F1 SETUP REQUEST message contains the gNB-DU Served Cells List IE, the gNB-CU shall take into account as specified in TS 38.401 [4].",
        "If the F1 SETUP RESPONSE message contains the BAP Address IE, the gNB-DU shall, if supported, store the received BAP address and use it as specified in TS 38.340 [30].",
        "If the TAI NSAG Support List IE is included in the Served Cell Information IE in the F1 SETUP REQUEST message, the gNB-CU shall, if supported, use this information as specified in TS 23.501 [21].",
        "The gNB-CU shall perform RRC Reconfiguration or RRC connection resume as described in TS 38.331 [8].",
        "If the SRB To Be Setup List IE is contained in the UE CONTEXT SETUP REQUEST message, the gNB-DU shall act as specified in TS 38.401 [4].",
        "If the DRB To Be Setup List IE is contained in the UE CONTEXT SETUP REQUEST message, the gNB-DU shall act as specified in TS 38.401 [4].",
        "The gNB-DU shall use the mapping information stored when forwarding traffic on BAP sublayer, as specified in TS 38.340 [30].",
        "The gNB-CU and gNB-DU use the Additional PDCP Duplication UP TNL Information IEs to support packet duplication for intra-gNB-DU CA as defined in TS 38.470 [2].",
        "The gNB-DU shall store the received Subscriber Profile ID for RAT/Frequency priority in the UE context and use it as defined in TS 36.300 [20].",
        "If the CG-ConfigInfo IE is included in the UE CONTEXT SETUP REQUEST message, the gNB-DU shall regard it as a reconfiguration with sync as defined in TS 38.331 [8].",
        "If the HandoverPreparationInformation IE is included in the CU to DU RRC Information IE in the UE CONTEXT SETUP REQUEST message, the gNB-DU shall regard it as a reconfiguration with sync as defined in TS 38.331 [8].",
        "If the NeedForGapNCSG-InfoEUTRA IE is included in the CU to DU RRC Information IE in the UE CONTEXT SETUP REQUEST message, the gNB-DU shall, if supported, use it as described in TS 38.331 [8].",
        "If the Trace Activation IE is included in the UE CONTEXT SETUP REQUEST message the gNB-DU shall, if supported, initiate the requested trace function as described in TS 32.422 [29].",
        "If the Configured BAP Address IE is contained in the UE CONTEXT SETUP REQUEST message, the gNB-DU shall, if supported, use it as specified in TS 38.340 [30].",
        "If the MDT Polluted Measurement Indicator IE is included in the UE CONTEXT SETUP REQUEST, the gNB-DU shall take this information into account as specified in TS 38.401 [4].",
        "If the SCG Activation Request IE is included in the UE CONTEXT SETUP REQUEST message, the gNB-DU may use it to configure SCG resources as specified in TS 37.340 [7].",
        "If the UL PDU Session Aggregate Maximum Bit Rate IE is included in the QoS Flow Level QoS Parameters IE contained in the UE CONTEXT MODIFICATION REQUEST message, the gNB-DU shall use it as specified in TS 23.501 [21].",
        "If the Resource Coordination Transfer Container IE is included in the UE CONTEXT MODIFICATION RESPONSE, the gNB-CU shall transparently transfer this information for the purpose of resource coordination as described in TS 38.423 [28].",
        "If the DU to CU RRC Information IE is included in the UE CONTEXT MODIFICATION RESPONSE message, the gNB-CU shall perform RRC Reconfiguration as described in TS 38.331 [8].",
        "If the Bearer Type Change IE is included in DRB to Be Modified List IE in the UE CONTEXT MODIFICATION REQUEST message, the gNB-DU shall reset the lower layers or generate a new LCID as specified in TS 37.340 [7].",
        "If the InterFrequencyConfig-NoGap IE is included in the DU to CU RRC Information IE contained in the UE CONTEXT MODIFICATION RESPONSE message, the gNB-CU shall, if supported, use it as described in TS 38.331 [8].",
        "If the Positioning Assistance Information IE is included in the POSITIONING ASSISTANCE INFORMATION CONTROL message, the gNB-DU shall use it as specified in TS 38.455 [37].",
        "If the Paging DRX IE is included in the MULTICAST GROUP PAGING message gNB-DU shall use it according to TS 38.304 [24].",
        "If the PEIPS Assistance Information IE is included in the PAGING message, the gNB-DU shall use it as specified in TS 38.300 [6].",
        "If the UEID Subgrouping Support Indication IE is included in the PAGING message, the gNB-DU shall use it for paging subgrouping as specified in TS 38.300 [6].",
        "The NR Paging eDRX Information for RRC INACTIVE IE may be included in the PAGING message, and the gNB-DU shall use it according to TS 38.304 [24].",
        "The gNB-DU shall not include the ReconfigurationWithSync field in the CellGroupConfig IE as defined in TS 38.331 [8]."
    ]
    ngap_sentences = [
        "For each PDU session, if the Network Instance IE is included in the PDU Session Resource Setup Request Transfer IE contained in the PDU SESSION RESOURCE SETUP REQUEST message and the Common Network Instance IE is not present, the NG-RAN node shall, if supported, use it when selecting transport network resource as specified in TS 23.501 [9].",
        
        "For each PDU session, if the Redundant UL NG-U UP TNL Information IE is included in the PDU Session Resource Setup Request Transfer IE of the PDU SESSION RESOURCE SETUP REQUEST message, the NG-RAN node shall, if supported, use it as the uplink termination point for the user plane data for this PDU session for the redundant transmission and it shall include the Redundant QoS Flow per TNL Information IE in the PDU Session Resource Setup Response Transfer IE as described in TS 23.501 [9].",
        
        "For each PDU session for which the Maximum Integrity Protected Data Rate Downlink IE or the Maximum Integrity Protected Data Rate Uplink IE are included in the Security Indication IE in the PDU Session Resource Setup Request Transfer IE of the PDU SESSION RESOURCE SETUP REQUEST message, the NG-RAN node shall store the respective information and, if integrity protection is to be performed for the PDU session, it shall enforce the traffic limits corresponding to the received values, for the concerned PDU session and concerned UE, as specified in TS 23.501 [9].",
        
        "Upon reception of the PDU SESSION RESOURCE SETUP REQUEST message to setup a QoS flow for IMS voice, if the NG-RAN node is not able to support IMS voice, the NG-RAN node shall initiate EPS fallback or RAT fallback for IMS voice procedure as specified in TS 23.501 [9] and report unsuccessful establishment of the QoS flow in the PDU Session Resource Setup Response Transfer IE or in the PDU Session Resource Setup Unsuccessful Transfer IE with cause value \"IMS voice EPS fallback or RAT fallback triggered\".",
        
        "For each PDU session, if the Redundant PDU Session Information IE is included in the PDU Session Resource Setup Request Transfer IE contained in the PDU SESSION RESOURCE SETUP REQUEST message, the NG-RAN node shall, if supported, store the received information in the UE context and setup the redundant user plane for the redundant PDU session as specified in TS38.300 [8] and TS 23.501 [9].",
        
        "For each PDU session, if the Redundant QoS Flow Indicator IE is included and set to \"false\" for all QoS flows, the NG-RAN node shall, if supported, stop the redundant transmission and release the redundant tunnel for the concerned PDU session as specified in TS 23.501 [9].",
        
        "Upon reception of the PDU SESSION RESOURCE MODIFY REQUEST message to setup a QoS flow for IMS voice, if the NG-RAN node is not able to support IMS voice, the NG-RAN node shall initiate EPS fallback or RAT fallback for IMS voice procedure as specified in TS 23.501 [9] and report unsuccessful establishment of the QoS flow in the PDU Session Resource Modify Response Transfer IE or in the PDU Session Resource Modify Unsuccessful Transfer IE with cause value \"IMS voice EPS fallback or RAT fallback triggered\".",
        
        "If the Mobility Restriction List IE is not contained in the INITIAL CONTEXT SETUP REQUEST message, the NG-RAN node shall consider that no roaming and no access restriction apply to the UE except for the PNI NPN mobility as described in TS 23.501 [9].",
        
        "The NG-RAN node shall consider that roaming or access to CAG cells is only allowed if the Allowed PNI-NPN List IE is contained in the INITIAL CONTEXT SETUP REQUEST message, as described in TS 23.501 [9].",
        
        "If the Trace Activation IE is included in the INITIAL CONTEXT SETUP REQUEST message the NG-RAN node shall, if supported, initiate the requested trace function as described in TS 32.422 [11].",
        
        "In particular, the NG-RAN node shall, if supported: if the Trace Activation IE includes the MDT Activation IE set to \"Immediate MDT and Trace\", initiate the requested trace session and MDT session as described in TS 32.422 [11] - if the Trace Activation IE includes the MDT Activation IE set to \"Immediate MDT Only\", \"Logged MDT only\", initiate the requested MDT session as described in TS 32.422 [11] and the NG-RAN node shall ignore the Interfaces To Trace IE and the Trace Depth IE - if the Trace Activation IE includes the MDT Location Information IE within the MDT Configuration IE, store this information and take it into account in the requested MDT session - if the Trace Activation IE includes the Signalling Based MDT PLMN List IE within the MDT Configuration IE, the NG-RAN node may use it to propagate the MDT Configuration as described in TS 37.320 [41].",
        
        "If the UE Security Capabilities IE included in the INITIAL CONTEXT SETUP REQUEST message only contains the EIA0 or NIA0 algorithm as defined in TS 33.501 [13] and if the EIA0 or NIA0 algorithm is defined in the configured list of allowed integrity protection algorithms in the NG-RAN node (TS 33.501 [13]), the NG-RAN node shall take it into use and ignore the keys received in the Security Key IE.",
        
        "If the Core Network Assistance Information for RRC INACTIVE IE is included in the INITIAL CONTEXT SETUP REQUEST message, the NG-RAN node shall, if supported, store this information in the UE context and use it for the RRC_INACTIVE state decision and RNA configuration for the UE and RAN paging if any for a UE in RRC_INACTIVE state, as specified in TS 38.300 [8].",
        
        "If the Emergency Fallback Indicator IE is included in the INITIAL CONTEXT SETUP REQUEST message, it indicates that the UE context to be set up is subject to emergency service fallback as described in TS 23.501 [9] and the NG-RAN node may, if supported, take the appropriate mobility actions.",
        
        "If the CE-mode-B Restricted IE is included in the INITIAL CONTEXT SETUP REQUEST message and the Enhanced Coverage Restriction IE is not set to \"restricted\" and the Enhanced Coverage Restriction information stored in the UE context is not set to \"restricted\", the NG-RAN node shall, if supported, store this information in the UE context and use it as defined in TS 23.501 [9].",
        
        "If the supported algorithms for encryption defined in the Encryption Algorithms IE in the UE Security Capabilities IE, plus the mandated support of EEA0 and NEA0 in all UEs (TS 33.501 [13]), do not match any allowed algorithms defined in the configured list of allowed encryption algorithms in the NG-RAN node (TS 33.501 [13]), the NG-RAN node shall reject the procedure using the INITIAL CONTEXT SETUP FAILURE message.",
        
        "If the supported algorithms for integrity defined in the Integrity Protection Algorithms IE in the UE Security Capabilities IE, plus the mandated support of the EIA0 and NIA0 algorithm in all UEs (TS 33.501 [13]), do not match any allowed algorithms defined in the configured list of allowed integrity protection algorithms in the NG-RAN node (TS 33.501 [13]), the NG-RAN node shall reject the procedure using the INITIAL CONTEXT SETUP FAILURE message.",
        
        "If the Security Key IE is included in the UE CONTEXT MODIFICATION REQUEST message, the NG-RAN node shall store it and perform AS key re-keying according to TS 33.501 [13].",
        
        "If the Emergency Fallback Indicator IE is included in the UE CONTEXT MODIFICATION REQUEST message, it indicates that the concerned UE context is subject to emergency service fallback as described in TS 23.501 [9] and the NG-RAN node may, if supported, take the appropriate mobility actions taking into account the Emergency Service Target CN IE if provided.",
        
        "If the UE Slice Maximum Bit Rate List IE is included in the UE CONTEXT MODIFICATION REQUEST message, the NG-RAN node shall, if supported: store and replace the previously provided UE Slice Maximum Bit Rate List, if any, by the received UE Slice Maximum Bit Rate List in the UE context - use the received UE Slice Maximum Bit Rate List for each S-NSSAI for the concerned UE as specified in TS 23.501 [9].",
        
        "If the CE-mode-B Restricted IE is included in the CONNECTION ESTABLISHMENT INDICATION message and the Enhanced Coverage Restriction IE is not set to \"restricted\" and the Enhanced Coverage Restricted information stored in the UE context is not set to \"restricted\", the NG-RAN node shall, if supported, store this information in the UE context and use it as defined in TS 23.501 [9].",
        
        "If the Security Context IE is included in the UE CONTEXT SUSPEND RESPONSE message, the NG-RAN node shall store the received Security Context IE in the UE context and remove any existing unused stored {NH, NCC} as specified in TS 33.501 [13].",
        
        "If the Security Context IE is included in the UE CONTEXT RESUME RESPONSE message, the NG-RAN node shall store the received Security Context IE in the UE context and the NG-RAN node shall use it for the next suspend/resume or Xn handover or Intra NG-RAN node handovers as specified in TS 33.501 [13].",
        
        "Upon reception of the UE CONTEXT RESUME FAILURE message the NG-RAN node shall release the RRC connection as specified in TS 36.331 [21] and release all related signalling and user data transport resources.",
        
        "In case of inter-system handover to LTE, the information in the Source to Target Transparent Container IE shall be encoded according to the Source eNB to Target eNB Transparent Container IE definition as specified in TS 36.413 [16].",
        
        "If the HANDOVER COMMAND message contains the QoS Flow to be Forwarded List IE and/or Data Forwarding Response DRB List IE within the Handover Command Transfer IE for a given PDU session, then the source NG-RAN node should initiate data forwarding for the QoS flows as specified in TS 38.300 [8].",
        
        "If the NAS Security Parameters from NG-RAN IE is included in the HANDOVER COMMAND message the NG-RAN node shall use it as specified in TS 33.501 [13].",
        
        "If the DAPS Request Information IE is included for a DRB in the Source NG-RAN Node to Target NG-RAN Node Transparent Container IE within the HANDOVER REQUIRED message, it indicates that the request concerns a DAPS Handover for that DRB, as described in TS 38.300 [8].",
        
        "If the HANDOVER REQUEST message contains the Redundant PDU Session Information IE associated with a given PDU session within the Handover Request Transfer IE, the target NG-RAN node shall, if supported, store the received information in the UE context and use it for redundant PDU session setup as specified in TS38.300 [8] and TS 23.501 [9].",
        
        "In case of inter-system handover from E-UTRAN with direct forwarding, if the target NG-RAN node receives the SgNB UE X2AP ID IE in the Source NG-RAN Node to Target NG-RAN Node Transparent Container IE, it may use it for internal forwarding as described in TS 37.340 [32].",
        
        "If the Mobility Restriction List IE is not contained in the HANDOVER REQUEST message, the target NG-RAN node shall consider that no roaming and no access restriction apply to the UE except for the PNI NPN mobility as described in TS 23.501 [9].",
        
        "The NG-RAN node shall consider that roaming or access to CAG cells is only allowed if the Allowed PNI-NPN List IE is contained in the HANDOVER REQUEST message, as described in TS 23.501 [9].",
        
        "If the New Security Context Indicator IE is included in the HANDOVER REQUEST message, the target NG-RAN node shall use the information as specified in TS 33.501 [13].",
        
        "If the NASC IE is included in the HANDOVER REQUEST message, the target NG-RAN node shall use it towards the UE as specified in TS 33.501 [13].",
        
        "If the supported algorithms for encryption defined in the Encryption Algorithms IE in the UE Security Capabilities IE, plus the mandated support of EEA0 and NEA0 in all UEs (TS 33.501 [13]), do not match any allowed algorithms defined in the configured list of allowed encryption algorithms in the NG-RAN node (TS 33.501 [13]), the target NG-RAN node shall reject the procedure using the HANDOVER FAILURE message.",
        
        "If the supported algorithms for integrity defined in the Integrity Protection Algorithms IE in the UE Security Capabilities IE, plus the mandated support of the EIA0 and NIA0 algorithm in all UEs (TS 33.501 [13]), do not match any allowed algorithms defined in the configured list of allowed integrity protection algorithms in the NG-RAN node (TS 33.501 [13]), the target NG-RAN node shall reject the procedure using the HANDOVER FAILURE message.",
        
        "For each PDU session for which the User Plane Security Information IE is included in the Path Switch Request Transfer IE of the PATH SWITCH REQUEST message, the SMF shall behave as specified in TS 33.501 [13] and may send back the Security Indication IE within the Path Switch Request Acknowledge Transfer IE of the PATH SWITCH REQUEST ACKNOWLEDGE message.",
        
        "If the Security Indication IE is included within the Path Switch Request Acknowledge Transfer IE of the PATH SWITCH REQUEST ACKNOWLEDGE message, the NG-RAN node shall behave as specified in TS 33.501 [13].",
        
        "If the New Security Context Indicator IE is included in the PATH SWITCH REQUEST ACKNOWLEDGE message, the NG-RAN node shall use the information as specified in TS 33.501 [13].",
        
        "Upon reception of the PATH SWITCH REQUEST ACKNOWLEDGE message the NG-RAN node shall store the received Security Context IE in the UE context and the NG-RAN node shall use it as specified in TS 33.501 [13].",
        
        "If the UE Security Capabilities IE is included in the PATH SWITCH REQUEST ACKNOWLEDGE message, the NG-RAN node shall handle it accordingly (TS 33.501 [13]).",
        
        "If the CE-mode-B Restricted IE is included in the PATH SWITCH REQUEST ACKNOWLEDGE message and the Enhanced Coverage Restriction IE is not set to \"restricted\" and the Enhanced Coverage Restriction information stored in the UE context is not set to \"restricted\", the NG-RAN node shall, if supported, store this information in the UE context and use it as defined in TS 23.501 [9].",
        
        "If the Mobility Restriction List IE is not contained in the DOWNLINK NAS TRANSPORT message and there is no previously stored mobility restriction information, the NG-RAN node shall consider that no roaming and no access restriction apply to the UE except for the PNI NPN mobility as described in TS 23.501 [9].",
        
        "The NG-RAN node shall consider that roaming or access to CAG cells is only allowed if the Allowed PNI-NPN List IE is contained in the DOWNLINK NAS TRANSPORT message, as described in TS 23.501 [9].",
        
        "If the CE-mode-B Restricted IE is included in the DOWNLINK NAS TRANSPORT message and the Enhanced Coverage Restriction IE is not set to \"restricted\" and the Enhanced Coverage Restricted information stored in the UE context is not set to \"restricted\", the NG-RAN node shall, if supported, store this information in the UE context and use it as defined in TS 23.501 [9].",
        
        "The NG-RAN node shall, if supported, reroute the INITIAL UE MESSAGE message to an AMF indicated by the AMF Set ID IE as described in TS 23.501 [9].",
        
        "Upon receipt of the AMF STATUS INDICATION message, the NG-RAN node shall consider the indicated GUAMI(s) will be unavailable and perform AMF reselection as defined in TS 23.501 [9].",
        
        "If the Warning Type IE is included in the WRITE-REPLACE WARNING REQUEST message, the NG-RAN node shall broadcast the Primary Notification irrespective of the setting of the Repetition Period IE and the Number of Broadcasts Requested IE, and process the Primary Notification according to TS 36.331 [21] and TS 38.331 [18].",
        
        "If the Trace Activation IE is included in the TRACE START message which includes the MDT Activation IE set to \"Immediate MDT and Trace\", the NG-RAN node shall, if supported, initiate the requested trace session and MDT session as described in TS 32.422 [11].",
        
        "Upon receipt of the UE RADIO CAPABILITY CHECK REQUEST message, the NG-RAN node checks whether the UE radio capabilities are compatible with the network configuration for IMS voice, and responds with a UE RADIO CAPABILITY CHECK RESPONSE message, as defined in TS 23.502 [10].",
        
        "If the Handover Flag IE is included in the SECONDARY RAT DATA USAGE REPORT message, it indicates that for each PDU session the AMF should buffer the Secondary RAT Data Usage Report Transfer IE since the secondary RAT data usage report is sent due to handover as defined in TS 23.502 [10].",
        
        "If unicast shared NG-U transport is used, the NG-RAN node shall include the Shared NG-U TNL Information IE in the MBS Distribution Release Request Transfer IE in the DISTRIBUTION RELEASE REQUEST message, and the MB-SMF shall release the corresponding shared NG-U transport as specified in TS 23.247 [44]."
    ]
    run_batch(ngap_sentences)