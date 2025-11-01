from typing import Dict, List, Optional
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

@dataclass
class FigureMetadata:
    """Metadata for an extracted figure."""
    caption: str
    file_path: Path
    file_type: str  # e.g., 'png', 'emf', 'wmf', 'drawingml'
    original_index: int  # Its order in the document
    r_id: str # The relationship ID from the docx XML
    target_ref: str # The target reference path from the docx XML
    
    # OLE Object metadata (added 2025-10-31)
    ole_prog_id: Optional[str] = None  # e.g., 'Visio.Drawing.15', 'Word.Picture.8'
    ole_object_path: Optional[Path] = None  # Path to extracted OLE object
    is_ole_object: bool = False  # True if figure is from OLE object
    nesting_level: int = 0  # Depth of nested extraction (0=direct, 1=nested, etc.)

@dataclass
class DocumentSection:
    """Represents a section of a 3GPP document."""
    title: str
    text: str
    clause: str
    document: str
    has_figure: bool = False
    figures: List['FigureMetadata'] = field(default_factory=list)
    is_procedure: bool = False
    procedure_name: Optional[str] = None

@dataclass
class Entity:
    """Enhanced entity with search capabilities."""
    name: str
    entity_type: str
    properties: Dict[str, any] = field(default_factory=dict)
    
    # NEW: Search-related fields (Requirement 8)
    description: Optional[str] = None
    parent_title: Optional[str] = None
    search_keywords: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    title_embedding: Optional[List[float]] = None
    parent_title_embedding: Optional[List[float]] = None
    
    def __post_init__(self):
        """Generate search keywords from name and description."""
        keywords = set()
        
        # Add name tokens
        keywords.update(self.name.lower().split())
        
        # Add description tokens if available
        if self.description:
            keywords.update(self.description.lower().split())

        # Add parent title tokens if available
        if self.parent_title:
            keywords.update(self.parent_title.lower().split())
        
        # Add entity type
        keywords.add(self.entity_type.lower())
        
        # Add procedure name if available
        if 'procedure' in self.properties:
            keywords.update(self.properties['procedure'].lower().split())
        
        self.search_keywords = list(keywords)
    
    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary for database storage."""
        return {
            'name': self.name,
            'entity_type': self.entity_type,
            'properties': self.properties,
            'description': self.description,
            'search_keywords': self.search_keywords
        }

@dataclass
class Relationship:
    """Enhanced relationship with FSM transition support."""
    source_name: str
    target_name: str
    rel_type: str
    properties: Dict[str, any] = field(default_factory=dict)
    
    # NEW: FSM-related fields (Requirement 10)
    is_transition: bool = False
    transition_condition: Optional[str] = None
    transition_action: Optional[str] = None
    
    def __post_init__(self):
        """Identify if this relationship represents an FSM transition."""
        # FOLLOWED_BY relationships are FSM transitions
        if self.rel_type == "FOLLOWED_BY":
            self.is_transition = True
            self.transition_condition = f"complete_{self.source_name}"
            self.transition_action = f"execute_{self.target_name}"
    
    def get_unique_id(self) -> str:
        """Generate unique ID for relationship."""
        content = f"{self.source_name}:{self.target_name}:{self.rel_type}"
        return hashlib.md5(content.encode()).hexdigest()

@dataclass
class ProcedureContext:
    """Enhanced procedure context with step descriptions."""
    procedure_name: str
    section: DocumentSection
    network_functions: List[str] = field(default_factory=list)
    messages: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    keys: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    
    # NEW: Step descriptions (Requirement 9)
    step_descriptions: Dict[str, str] = field(default_factory=dict)
    
    # NEW: Search metadata (Requirement 8)
    search_description: Optional[str] = None
    search_tags: List[str] = field(default_factory=list)

@dataclass
class ExtractionResult:
    """Result of entity/relationship extraction."""
    entities: Dict[str, List[str]] = field(default_factory=dict)
    relationships: List[Relationship] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    
    # NEW: Enhanced extraction metadata
    extraction_confidence: float = 1.0
    extraction_method: str = "unknown"
    llm_model_used: Optional[str] = None

# NEW: FSM Support Classes (Requirement 10)
@dataclass
class FSMState:
    """Represents a state in the finite state machine."""
    name: str
    step_entity: Entity
    is_initial: bool = False
    is_final: bool = False
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'name': self.name,
            'is_initial': self.is_initial,
            'is_final': self.is_final,
            'description': self.description,
            'step_entity_name': self.step_entity.name
        }

@dataclass
class FSMTransition:
    """Represents a transition in the finite state machine."""
    source_state: str
    target_state: str
    trigger: str
    condition: Optional[str] = None
    action: Optional[str] = None
    message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'source_state': self.source_state,
            'target_state': self.target_state,
            'trigger': self.trigger,
            'condition': self.condition,
            'action': self.action,
            'message': self.message
        }

@dataclass
class FiniteStateMachine:
    """Complete FSM representation of a procedure."""
    procedure_name: str
    states: List[FSMState] = field(default_factory=list)
    transitions: List[FSMTransition] = field(default_factory=list)
    initial_state: Optional[str] = None
    final_states: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, any]:
        return {
            'procedure_name': self.procedure_name,
            'states': [state.to_dict() for state in self.states],
            'transitions': [trans.to_dict() for trans in self.transitions],
            'initial_state': self.initial_state,
            'final_states': self.final_states
        }
    
    def to_json(self) -> str:
        """Export FSM as JSON for conformance testing tools."""
        return json.dumps(self.to_dict(), indent=2)

# NEW: Search Support Classes (Requirement 8)
@dataclass
class SearchQuery:
    """Query for searching entities and procedures."""
    query_text: str
    entity_types: Optional[List[str]] = None
    max_results: int = 10
    similarity_threshold: float = 0.3
    filters: Optional[Dict[str, any]] = None

@dataclass
class SearchResult:
    """Result from entity/procedure search."""
    entity: 'Entity'
    similarity_score: float
    match_type: str  # 'exact', 'tfidf', 'semantic', etc.
    matched_keywords: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, any]] = None