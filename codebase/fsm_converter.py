from typing import List, Dict, Optional
import json
from pathlib import Path
from datetime import datetime

from data_structures import (
    Entity, Relationship, FiniteStateMachine, FSMState, 
    FSMTransition, ProcedureContext
)
from config import FSM_CONFIG

class FSMConverter:
    """Convert knowledge graph procedures to Finite State Machines (Requirement 10)."""
    
    def __init__(self):
        print("FSMConverter initialized")
    
    def convert_procedure_to_fsm(self, 
                               procedure_name: str,
                               entities: List[Entity],
                               relationships: List[Relationship],
                               context: Optional[ProcedureContext] = None) -> FiniteStateMachine:
        """Convert a procedure from knowledge graph to FSM."""
        print(f"Converting procedure '{procedure_name}' to FSM...")
        
        # Find procedure-related entities
        procedure_entities = self._filter_procedure_entities(procedure_name, entities)
        procedure_relationships = self._filter_procedure_relationships(procedure_name, relationships)
        
        # Extract steps (which become FSM states)
        step_entities = [e for e in procedure_entities if e.entity_type == "Step"]
        step_entities.sort(key=lambda x: self._extract_step_number(x.name))
        
        # Create FSM states
        states = self._create_fsm_states(step_entities, context)
        
        # Create FSM transitions
        transitions = self._create_fsm_transitions(step_entities, procedure_relationships, entities)
        
        # Determine initial and final states
        initial_state = states[0].name if states else None
        final_states = [states[-1].name] if states else []
        
        fsm = FiniteStateMachine(
            procedure_name=procedure_name,
            states=states,
            transitions=transitions,
            initial_state=initial_state,
            final_states=final_states
        )
        
        print(f"✓ FSM created: {len(states)} states, {len(transitions)} transitions")
        return fsm
    
    def _filter_procedure_entities(self, procedure_name: str, entities: List[Entity]) -> List[Entity]:
        """Filter entities related to specific procedure."""
        procedure_entities = []
        
        for entity in entities:
            # Direct procedure match
            if entity.name == procedure_name:
                procedure_entities.append(entity)
                continue
            
            # Step entities for this procedure
            if (entity.entity_type == "Step" and 
                procedure_name.replace(' ', '_').lower() in entity.name.lower()):
                procedure_entities.append(entity)
                continue
            
            # Entities with procedure property
            if entity.properties.get('procedure') == procedure_name:
                procedure_entities.append(entity)
        
        return procedure_entities
    
    def _filter_procedure_relationships(self, procedure_name: str, relationships: List[Relationship]) -> List[Relationship]:
        """Filter relationships related to specific procedure."""
        procedure_relationships = []
        
        for rel in relationships:
            # Check if either source or target is related to the procedure
            if (procedure_name.replace(' ', '_').lower() in rel.source_name.lower() or
                procedure_name.replace(' ', '_').lower() in rel.target_name.lower() or
                rel.source_name == procedure_name or
                rel.target_name == procedure_name):
                procedure_relationships.append(rel)
        
        return procedure_relationships
    
    def _create_fsm_states(self, step_entities: List[Entity], context: Optional[ProcedureContext]) -> List[FSMState]:
        """Create FSM states from step entities."""
        states = []
        
        for i, step_entity in enumerate(step_entities):
            # Get step description
            description = None
            if context and step_entity.name in context.step_descriptions:
                description = context.step_descriptions[step_entity.name]
            elif step_entity.description:
                description = step_entity.description
            else:
                description = f"Execute step {i+1} of the procedure"
            
            state = FSMState(
                name=step_entity.name,
                step_entity=step_entity,
                is_initial=(i == 0),
                is_final=(i == len(step_entities) - 1),
                description=description
            )
            
            states.append(state)
        
        return states
    
    def _create_fsm_transitions(self, step_entities: List[Entity], 
                              relationships: List[Relationship],
                              all_entities: List[Entity]) -> List[FSMTransition]:
        """Create FSM transitions from relationships."""
        transitions = []
        
        # Create transitions from FOLLOWED_BY relationships
        followed_by_rels = [r for r in relationships if r.rel_type == "FOLLOWED_BY"]
        
        for rel in followed_by_rels:
            # Find trigger message if any
            trigger_message = self._find_trigger_message(rel, relationships, all_entities)
            
            transition = FSMTransition(
                source_state=rel.source_name,
                target_state=rel.target_name,
                trigger=trigger_message or "step_complete",
                condition=f"complete_{rel.source_name}",
                action=f"execute_{rel.target_name}",
                message=trigger_message
            )
            
            transitions.append(transition)
        
        return transitions
    
    def _find_trigger_message(self, step_relation: Relationship, 
                            all_relationships: List[Relationship],
                            all_entities: List[Entity]) -> Optional[str]:
        """Find message that triggers transition between steps."""
        # Look for messages sent in the source step
        for rel in all_relationships:
            if (rel.rel_type in ["SEND", "SEND_BY"] and 
                step_relation.source_name in [rel.source_name, rel.target_name]):
                
                # Find the message entity
                for entity in all_entities:
                    if (entity.entity_type == "Message" and 
                        entity.name in [rel.source_name, rel.target_name]):
                        return entity.name
        
        return None
    
    def _extract_step_number(self, step_name: str) -> int:
        """Extract step number from step name for sorting."""
        import re
        match = re.search(r'step_(\d+)', step_name)
        return int(match.group(1)) if match else 0
    
    def export_fsm_to_json(self, fsm: FiniteStateMachine, output_path: Path):
        """Export FSM to JSON file for conformance testing tools."""
        fsm_dict = fsm.to_dict()
        
        # Add conformance testing metadata (FIXED: removed Path.ctime())
        fsm_dict['metadata'] = {
            'format_version': '1.0',
            'generator': '3GPP Knowledge Graph Builder',
            'export_date': datetime.now().isoformat(),
            'conformance_ready': True
        }
        
        with open(output_path, 'w') as f:
            json.dump(fsm_dict, f, indent=2)
        
        print(f"✓ FSM exported to {output_path}")
    
    def export_fsm_to_dot(self, fsm: FiniteStateMachine, output_path: Path):
        """Export FSM to DOT format for visualization."""
        dot_content = self._generate_dot_content(fsm)
        
        with open(output_path, 'w') as f:
            f.write(dot_content)
        
        print(f"✓ FSM DOT file exported to {output_path}")
    
    def _generate_dot_content(self, fsm: FiniteStateMachine) -> str:
        """Generate DOT format content for FSM visualization."""
        lines = [
            "digraph FSM {",
            "  rankdir=TB;",
            "  node [shape=circle];",
            ""
        ]
        
        # Add states
        for state in fsm.states:
            shape = "doublecircle" if state.is_final else "circle"
            color = "green" if state.is_initial else "lightblue"
            
            lines.append(f'  "{state.name}" [shape={shape}, fillcolor={color}, style=filled];')
        
        lines.append("")
        
        # Add transitions
        for transition in fsm.transitions:
            label = transition.trigger
            if transition.message:
                label += f"\\n{transition.message}"
            
            lines.append(f'  "{transition.source_state}" -> "{transition.target_state}" [label="{label}"];')
        
        lines.append("}")
        
        return "\n".join(lines)
    
    def validate_fsm(self, fsm: FiniteStateMachine) -> Dict[str, bool]:
        """Validate FSM for conformance testing requirements."""
        validation_results = {
            'has_initial_state': fsm.initial_state is not None,
            'has_final_states': len(fsm.final_states) > 0,
            'states_connected': self._check_connectivity(fsm),
            'no_orphaned_states': self._check_no_orphans(fsm),
            'deterministic': self._check_deterministic(fsm)
        }
        
        all_valid = all(validation_results.values())
        validation_results['overall_valid'] = all_valid
        
        return validation_results
    
    def _check_connectivity(self, fsm: FiniteStateMachine) -> bool:
        """Check if all states are reachable from initial state."""
        if not fsm.initial_state:
            return False
        
        reachable = set([fsm.initial_state])
        changed = True
        
        while changed:
            changed = False
            for transition in fsm.transitions:
                if transition.source_state in reachable and transition.target_state not in reachable:
                    reachable.add(transition.target_state)
                    changed = True
        
        all_states = set(state.name for state in fsm.states)
        return len(reachable) == len(all_states)
    
    def _check_no_orphans(self, fsm: FiniteStateMachine) -> bool:
        """Check that all states participate in transitions."""
        states_in_transitions = set()
        
        for transition in fsm.transitions:
            states_in_transitions.add(transition.source_state)
            states_in_transitions.add(transition.target_state)
        
        all_states = set(state.name for state in fsm.states)
        
        # Initial state might not have incoming transitions
        if fsm.initial_state:
            states_in_transitions.add(fsm.initial_state)
        
        return len(states_in_transitions) == len(all_states)
    
    def _check_deterministic(self, fsm: FiniteStateMachine) -> bool:
        """Check if FSM is deterministic (no ambiguous transitions)."""
        state_triggers = {}
        
        for transition in fsm.transitions:
            key = (transition.source_state, transition.trigger)
            if key in state_triggers:
                return False  # Multiple transitions with same trigger from same state
            state_triggers[key] = transition.target_state
        
        return True