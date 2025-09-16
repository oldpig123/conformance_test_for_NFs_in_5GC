from typing import Dict, List, Optional, Tuple
import json
import os
from pathlib import Path
from data_structures import FSMState, FSMTransition, FiniteStateMachine, Entity

class FSMConverter:
    """Convert 3GPP procedures to Finite State Machines for conformance testing."""
    
    def __init__(self):
        print("FSMConverter initialized")
        # Ensure output directory exists
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
    
    def convert_procedure_to_fsm(self, procedure_name: str, step_entities: List[Entity], 
                               relationships: List[Dict]) -> FiniteStateMachine:
        """Convert a 3GPP procedure to FSM representation."""
        print(f"Converting procedure '{procedure_name}' to FSM...")
        
        if not step_entities:
            print(f"  ⚠️ No steps found for {procedure_name}")
            return None
        
        # Create states from steps
        states = []
        step_names = [entity.name for entity in step_entities]
        
        for i, step_entity in enumerate(step_entities):
            # Truncate the full description for the FSM state's own description field
            # FIX: Handle cases where the entity description is None.
            # The description can be None, so we provide a default empty string.
            short_description = step_entity.description if step_entity.description is not None else ""
            if len(short_description) > 100:
                short_description = short_description[:100] + "..."
            
            # Create FSM state with required step_entity
            fsm_state = FSMState(
                name=step_entity.name,
                step_entity=step_entity,  # REQUIRED parameter
                is_initial=(i == 0),      # Determine if it's the initial state
                is_final=(i == len(step_entities) - 1), # Determine if it's a final state
                description=short_description
            )
            states.append(fsm_state)
        
        # Create transitions from relationships
        transitions = []
        
        # Add sequential transitions by default
        for i in range(len(step_entities) - 1):
            current_step = step_entities[i].name
            next_step = step_entities[i + 1].name
            
            # Create transition using correct field names from data_structures.py
            transition = FSMTransition(
                source_state=current_step,
                target_state=next_step,
                trigger="step_complete",
                condition="",
                action="proceed_to_next_step"
            )
            transitions.append(transition)
        
        # Add message-based transitions from relationships
        message_transitions = self._extract_message_transitions(relationships, step_names)
        transitions.extend(message_transitions)
        
        # Create FSM using correct field name from data_structures.py
        fsm = FiniteStateMachine(
            procedure_name=procedure_name,  # Matches data_structures.py
            states=states,
            transitions=transitions,
            initial_state=step_entities[0].name if step_entities else None,
            final_states=[step_entities[-1].name] if step_entities else []
        )
        
        print(f"✓ FSM created: {len(states)} states, {len(transitions)} transitions")
        return fsm
    
    def _extract_message_transitions(self, relationships: List[Dict], step_names: List[str]) -> List[FSMTransition]:
        """Extract message-based transitions from relationships."""
        transitions = []
        
        # Group SEND relationships by step
        step_sends = {}
        for rel in relationships:
            if rel.get('rel_type') == 'SEND' and rel.get('source_name') in step_names:
                step = rel['source_name']
                message = rel['target_name']
                if step not in step_sends:
                    step_sends[step] = []
                step_sends[step].append(message)
        
        # Create transitions based on sends
        sorted_steps = sorted(step_names)
        for i, step in enumerate(sorted_steps[:-1]):
            next_step = sorted_steps[i + 1]
            
            # Get messages sent by current step
            messages = step_sends.get(step, [])
            
            if messages:
                # Use first message as trigger
                primary_message = messages[0]
                transition = FSMTransition(
                    source_state=step,
                    target_state=next_step,
                    trigger=primary_message,
                    condition="",
                    action=f"send_{primary_message.lower().replace(' ', '_')}"
                )
                transitions.append(transition)
        
        return transitions
    
    def validate_fsm(self, fsm: FiniteStateMachine) -> Tuple[bool, List[str]]:
        """Validate FSM structure and return validation results."""
        errors = []
        
        if not fsm.states:
            errors.append("FSM has no states")
        
        if not fsm.initial_state:
            errors.append("FSM has no initial state")
        
        if not fsm.final_states:
            errors.append("FSM has no final states")
        
        # Check if initial state exists
        state_names = [state.name for state in fsm.states]
        if fsm.initial_state and fsm.initial_state not in state_names:
            errors.append(f"Initial state '{fsm.initial_state}' not found in states")
        
        # Check if final states exist
        for final_state in fsm.final_states:
            if final_state not in state_names:
                errors.append(f"Final state '{final_state}' not found in states")
        
        # Check transition validity
        for transition in fsm.transitions:
            if transition.source_state not in state_names:
                errors.append(f"Transition from unknown state: {transition.source_state}")
            if transition.target_state not in state_names:
                errors.append(f"Transition to unknown state: {transition.target_state}")
        
        # Validate step entities (they are required in your data structure)
        for state in fsm.states:
            if not state.step_entity:
                errors.append(f"State '{state.name}' has no associated step entity")
            elif state.step_entity.entity_type != "Step":
                errors.append(f"State '{state.name}' entity is not of type 'Step'")
        
        # Check reachability
        if fsm.initial_state and not self._is_reachable(fsm):
            errors.append("Some states are not reachable from initial state")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _is_reachable(self, fsm: FiniteStateMachine) -> bool:
        """Check if all states are reachable from initial state."""
        if not fsm.initial_state or not fsm.transitions:
            return len(fsm.states) <= 1
        
        visited = set()
        to_visit = [fsm.initial_state]
        
        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)
            
            # Find transitions from current state
            for transition in fsm.transitions:
                if transition.source_state == current and transition.target_state not in visited:
                    to_visit.append(transition.target_state)
        
        state_names = [state.name for state in fsm.states]
        return len(visited) == len(state_names)
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by removing/replacing invalid characters."""
        # Replace invalid characters with underscores
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Replace multiple underscores with single underscore
        while '__' in filename:
            filename = filename.replace('__', '_')
        
        # Remove leading/trailing underscores and spaces
        filename = filename.strip('_ ')
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename
    
    def export_fsm_to_json(self, fsm: FiniteStateMachine, filename: str) -> bool:
        """Export FSM to JSON file with proper directory creation."""
        try:
            # Sanitize the filename
            safe_filename = self._sanitize_filename(filename)
            
            # Ensure it ends with .json
            if not safe_filename.endswith('.json'):
                safe_filename += '.json'
            
            # Create full path in output directory
            output_path = self.output_dir / safe_filename
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert FSM to dictionary using the built-in method
            fsm_dict = fsm.to_dict()
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(fsm_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✓ FSM exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export FSM to JSON: {e}")
            return False
    
    def export_fsm_to_dot(self, fsm: FiniteStateMachine, filename: str) -> bool:
        """Export FSM to DOT format for Graphviz visualization."""
        try:
            # Sanitize the filename
            safe_filename = self._sanitize_filename(filename)
            
            # Ensure it ends with .dot
            if not safe_filename.endswith('.dot'):
                if safe_filename.endswith('.json'):
                    safe_filename = safe_filename[:-5] + '.dot'
                else:
                    safe_filename += '.dot'
            
            # Create full path in output directory
            output_path = self.output_dir / safe_filename
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate DOT content
            dot_content = self._generate_dot_content(fsm)
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(dot_content)
            
            print(f"✓ FSM DOT file exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to export FSM to DOT: {e}")
            return False
    
    def _generate_dot_content(self, fsm: FiniteStateMachine) -> str:
        """Generate DOT format content for FSM visualization."""
        lines = [
            "digraph FSM {",
            "  rankdir=TB;",
            "  node [shape=box, style=rounded];",
            ""
        ]
        
        # Add states
        for state in fsm.states:
            shape = "doublecircle" if state.is_final else "box"
            color = "green" if state.is_initial else ("red" if state.is_final else "lightblue")
            
            # Get description from step_entity
            if state.step_entity and state.step_entity.description:
                desc = state.step_entity.description
            else:
                desc = state.description or ""
            
            # Truncate description for visualization
            if len(desc) > 50:
                desc = desc[:50] + "..."
            
            lines.append(f'  "{state.name}" [shape={shape}, style=filled, fillcolor={color}, '
                        f'label="{state.name}\\n{desc}"];')
        
        lines.append("")
        
        # Add transitions
        for transition in fsm.transitions:
            label = transition.trigger if transition.trigger else "ε"
            lines.append(f'  "{transition.source_state}" -> "{transition.target_state}" '
                        f'[label="{label}"];')
        
        lines.append("}")
        
        return "\n".join(lines)