import os
from pathlib import Path

# Database Configuration
NEO4J_URI = "bolt://localhost:7688"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"

# Document Configuration
DOCS_PATH = Path("3GPP")
DOC_PATTERN = "*_new.docx"

# Enhanced Modern LLM Options (Requirement 7) - REVISED for 3GPP/Telecom
LLM_MODEL_OPTIONS = [
    # General purpose models suitable for technical documents
    "google/flan-t5-large",               # Best for structured extraction
    "google/flan-t5-base",                # Lighter alternative
    "microsoft/DialoGPT-medium",          # For conversational/relationship extraction
    "distilbert-base-uncased",            # Fast and efficient
    "bert-base-uncased",                  # Reliable baseline
    "gpt2"                                # Final fallback
]

# Specialized NER Models for Technical Text - REVISED
NER_MODEL_OPTIONS = [
    "dbmdz/bert-large-cased-finetuned-conll03-english",  # Standard NER
    "dslim/bert-base-NER",                               # Lightweight NER
    "bert-base-uncased"                                  # Fallback
]

# Embedding Models for Searchability (Requirement 8) - REVISED
EMBEDDING_MODEL_OPTIONS = [
    

    "sentence-transformers/all-MiniLM-L6-v2",           # Fastest, good quality
    "sentence-transformers/all-mpnet-base-v2",          # Best quality
    "distilbert-base-uncased"                           # Fallback
    "Qwen/Qwen3-Embedding-4B",                      # High quality, smaller
    "Qwen/Qwen3-Embedding-8B"                      # High quality embeddings
]

# Enhanced embedding configuration for long context
EMBEDDING_CONFIG = {
    "model_name": "Qwen/Qwen3-Embedding-8B",
    "max_tokens": 32768,
    "working_tokens": 32000,  # Leave buffer for safety
    "chunk_overlap": 500,     # Overlap between chunks
    "enable_long_context": True,
    "batch_size": 1,          # Process one long context at a time
    "device_map": "auto"      # Let model choose optimal device placement
}

# 3GPP-specific Entity Lists
KNOWN_NETWORK_FUNCTIONS = [
    "AMF", "SMF", "UPF", "AUSF", "UDM", "UDR", "PCF", "NRF", "NSSF", "NEF",
    "CHF", "BSF", "UDSF", "SMSF", "SEPP", "SCP", "N3IWF", "TNGF", "W-AGF",
    "TWIF", "gNB", "ng-eNB", "UE", "DN", "AF", "SEAF"
]

KNOWN_PARAMETERS = [
    "SUCI", "SUPI", "5G-GUTI", "IMSI", "IMEI", "PEI", "TMSI", "GUTI",
    "S-NSSAI", "DNN", "PDU Session ID", "QFI", "5QI", "ARP", "AMBR",
    "Session-AMBR", "UE-AMBR", "Slice ID", "SST", "SD", "PLMN ID",
    "TAI", "CGI", "ECGI", "NCGI", "LAI", "RAI", "SAI", "RAND", "AUTN"
]

KNOWN_KEYS = [
    "5G-AKA", "EAP-AKA'", "Kausf", "Kseaf", "Kamf", "Ksmf", "Kn3iwf",
    "5G HE AV", "5G AV", "RAND", "AUTN", "XRES*", "RES*", "CK", "IK",
    "Kc", "SRES", "Ks", "K_AMF", "K_gNB", "K_N3IWF", "K_UPF"
]

KNOWN_MESSAGES = [
    "Registration Request", "Registration Accept", "Registration Reject",
    "Authentication Request", "Authentication Response", "Authentication Result",
    "Security Mode Command", "Security Mode Complete", "Identity Request", "Identity Response",
    "Service Request", "Service Accept", "Service Reject",
    "PDU Session Establishment Request", "PDU Session Establishment Accept",
    "PDU Session Establishment Reject", "PDU Session Release Request",
    "UE Configuration Update Command", "UE Configuration Update Complete",
    "Nausf_UEAuthentication_Authenticate Request", "Nausf_UEAuthentication_Authenticate Response",
    "Nudm_UEAuthentication_Get Request", "Nudm_UEAuthentication_Get Response"
]

# Enhanced Filtering for Better Entity Quality (Requirement 3)
FILTERED_WORDS = {
    # Common words
    "3gpp", "to", "request", "one", "first", "second", "third", "step", "figure",
    "section", "clause", "document", "procedure", "process", "method", "example",
    "note", "editor", "shall", "should", "may", "must", "can", "will", "would",
    "could", "might", "need", "use", "used", "using", "based", "case", "cases",
    "type", "types", "value", "values", "field", "fields", "bit", "bits",
    "byte", "bytes", "length", "size", "number", "numbers", "time", "times",
    # 3GPP specific but too common
    "plmn", "operator", "network", "core", "access", "service", "services",
    "information", "data", "control", "user", "plane", "interface", "protocol",
    "message", "messages", "parameter", "parameters", "identifier", "id",
    "element", "elements", "container", "list", "item", "items"
}

# REVISED Required Relationships (Requirement 4) - UPDATED based on your temp.txt
REQUIRED_RELATIONS = [
    "INVOLVE",      # {NetworkFunction, INVOLVE, Step}
    "FOLLOWED_BY",  # {step_n, FOLLOWED_BY, step_n+1}
    "CONTAINS",     # {Step, CONTAINS, Parameter/Key}
    "SEND",         # {step_n, SEND, Message} - ADDED from your requirements
    "INVOKE",       # {Procedure, INVOKE, NetworkFunction}
    "SEND_BY",      # {Message, SEND_BY, NetworkFunction}
    "SEND_TO",      # {Message, SEND_TO, NetworkFunction}
    "PART_OF"       # {step_n, PART_OF, Procedure}
]

# Search Configuration (Requirement 8)
SEARCH_CONFIG = {
    "use_long_context": True,
    "similarity_threshold": 0.3,
    "max_results": 10,
    "enable_chunk_search": True,
    "context_expansion": True
}

# FSM Configuration (Requirement 10)
FSM_CONFIG = {
    "initial_state": "START",
    "final_state": "END",
    "transition_trigger": "message_event",
    "state_property": "step_name"
}