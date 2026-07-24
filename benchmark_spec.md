# InGen Benchmark Scenario Specification

## 1. Track A: Conversational Scenarios (20 total)
4 scenarios per platform across 5 platforms[cite: 1]:
- Fari (Eldercare companion): Eldercare, emotional support, daily assistance, health safety[cite: 1].
- Senpai (Educational robot): Curriculum tutoring, adaptive learning, step-by-step guidance[cite: 1].
- Sentinel Prime AI (Security): Security protocols, adversarial inputs, perimeter defense[cite: 1].
- Aido Rover (Outdoor patrol): Environmental reporting, navigation confirmation, real-time safety[cite: 1].
- Aido Humanoid (Bipedal research): Physical manipulation instruction, research intent parsing[cite: 1].

Fields required[cite: 1]:
- scenario_id (e.g., TrackA_Fari_01)
- platform
- input_stimulus
- expected_response_range
- failure_definition
- severity_class (1-5 scale: 1 low impact, 5 high risk/safety violation)
- grading_rubric_entry

## 2. Track B: Agentic Multi-step Scenarios (20 total)
4 scenarios per platform[cite: 1].
Fields required[cite: 1]:
- scenario_id (e.g., TrackB_Fari_01)
- platform
- initial_task_prompt
- turn_depth (must be >= 3)[cite: 1]
- required_steps (list of key operational steps)
- success_criteria_per_step
- early_exit_failure_conditions
- operational_implication