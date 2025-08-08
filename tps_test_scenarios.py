#!/usr/bin/env python3
"""
TPS Test Scenarios Library - Comprehensive Testing Suite
Extensive collection of test scenarios for validating TPS reasoning systems
"""

import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

class TestCategory(Enum):
    BASIC_FUNCTIONALITY = "basic_functionality"
    EMOTIONAL_PROCESSING = "emotional_processing"
    LOGICAL_ANALYSIS = "logical_analysis"
    HOLDING_SPACE = "holding_space"
    INTEGRATION_QUALITY = "integration_quality"
    WAVE_PROGRESSION = "wave_progression"
    DOMAIN_SELECTION = "domain_selection"
    EDGE_CASES = "edge_cases"
    PERFORMANCE = "performance"
    THERAPEUTIC = "therapeutic"
    DECISION_MAKING = "decision_making"
    CREATIVE_FLOW = "creative_flow"
    CONFLICT_RESOLUTION = "conflict_resolution"
    CRISIS_INTERVENTION = "crisis_intervention"

class ExpectedOutcome(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    ERROR = "error"

@dataclass
class TestScenario:
    """Comprehensive test scenario definition"""
    id: str
    name: str
    category: TestCategory
    description: str
    input_text: str
    expected_outcomes: Dict[str, Any]
    success_criteria: Dict[str, float]
    tags: List[str] = field(default_factory=list)
    difficulty_level: int = 3  # 1-5 scale
    estimated_processing_time: float = 5.0  # seconds
    configuration_requirements: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

class TPSTestScenarioLibrary:
    """Comprehensive TPS test scenario library"""
    
    def __init__(self):
        self.scenarios = {}
        self.categories = {}
        self.performance_benchmarks = {}
        self.initialize_scenarios()
    
    def initialize_scenarios(self):
        """Initialize all test scenarios"""
        
        # Basic Functionality Tests
        self.add_basic_functionality_tests()
        
        # Emotional Processing Tests
        self.add_emotional_processing_tests()
        
        # Logical Analysis Tests
        self.add_logical_analysis_tests()
        
        # Holding Space Tests
        self.add_holding_space_tests()
        
        # Integration Quality Tests
        self.add_integration_quality_tests()
        
        # Wave Progression Tests
        self.add_wave_progression_tests()
        
        # Domain Selection Tests
        self.add_domain_selection_tests()
        
        # Edge Case Tests
        self.add_edge_case_tests()
        
        # Performance Tests
        self.add_performance_tests()
        
        # Specialized Application Tests
        self.add_therapeutic_tests()
        self.add_decision_making_tests()
        self.add_creative_flow_tests()
        self.add_conflict_resolution_tests()
        self.add_crisis_intervention_tests()
    
    def add_basic_functionality_tests(self):
        """Basic functionality test scenarios"""
        
        scenarios = [
            TestScenario(
                id="basic_001",
                name="Simple Emotional Query",
                category=TestCategory.BASIC_FUNCTIONALITY,
                description="Test basic emotional processing with simple input",
                input_text="I'm feeling sad today and don't know why",
                expected_outcomes={
                    "dominant_sense": "E",
                    "domain": "psychology",
                    "wave_stages_completed": 5,
                    "integration_quality": 0.6
                },
                success_criteria={
                    "overall_success": 0.7,
                    "emotional_score": 6.0,
                    "wave_completion": 1.0
                },
                tags=["basic", "emotional", "simple"],
                difficulty_level=2
            ),
            
            TestScenario(
                id="basic_002",
                name="Simple Logical Query",
                category=TestCategory.BASIC_FUNCTIONALITY,
                description="Test basic logical processing with analytical input",
                input_text="I need to analyze which investment option would be better for my portfolio",
                expected_outcomes={
                    "dominant_sense": "L",
                    "domain": "physics",
                    "wave_stages_completed": 5,
                    "integration_quality": 0.7
                },
                success_criteria={
                    "overall_success": 0.75,
                    "logical_score": 7.0,
                    "wave_completion": 1.0
                },
                tags=["basic", "logical", "analysis"],
                difficulty_level=2
            ),
            
            TestScenario(
                id="basic_003",
                name="Simple Holding Query",
                category=TestCategory.BASIC_FUNCTIONALITY,
                description="Test basic holding space with uncertainty",
                input_text="I'm not sure what I want to do with my life right now, and that's okay",
                expected_outcomes={
                    "dominant_sense": "H",
                    "domain": "biology",
                    "wave_stages_completed": 5,
                    "integration_quality": 0.8
                },
                success_criteria={
                    "overall_success": 0.8,
                    "holding_score": 7.0,
                    "wave_completion": 1.0
                },
                tags=["basic", "holding", "uncertainty"],
                difficulty_level=2
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_emotional_processing_tests(self):
        """Emotional processing test scenarios"""
        
        scenarios = [
            TestScenario(
                id="emotion_001",
                name="Deep Grief Processing",
                category=TestCategory.EMOTIONAL_PROCESSING,
                description="Test handling of deep grief and loss",
                input_text="I lost my partner six months ago and I'm still struggling to get through each day. Sometimes I feel angry, sometimes numb, sometimes like I can't breathe. I don't know how to move forward without them.",
                expected_outcomes={
                    "dominant_sense": "E",
                    "domain": "psychology",
                    "emotional_score_range": [8.0, 10.0],
                    "response_style": "poetic",
                    "therapeutic_elements": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "emotional_validation": 0.9,
                    "holding_space_quality": 0.8,
                    "integration_quality": 0.7
                },
                tags=["grief", "loss", "trauma", "therapeutic"],
                difficulty_level=5,
                configuration_requirements=["therapeutic_support"]
            ),
            
            TestScenario(
                id="emotion_002",
                name="Anxiety Overwhelm",
                category=TestCategory.EMOTIONAL_PROCESSING,
                description="Test handling of anxiety and overwhelm",
                input_text="I feel like everything is spinning out of control. Work is overwhelming, my relationships are strained, and I can't sleep. My mind races constantly with worst-case scenarios.",
                expected_outcomes={
                    "dominant_sense": "E",
                    "domain": "psychology",
                    "emotional_score_range": [7.5, 9.5],
                    "grounding_elements": True,
                    "stabilization_focus": True
                },
                success_criteria={
                    "overall_success": 0.75,
                    "emotional_regulation": 0.8,
                    "practical_guidance": 0.7,
                    "wave_completion": 0.8
                },
                tags=["anxiety", "overwhelm", "stress", "stabilization"],
                difficulty_level=4
            ),
            
            TestScenario(
                id="emotion_003",
                name="Joy and Expansion",
                category=TestCategory.EMOTIONAL_PROCESSING,
                description="Test handling of positive emotions and expansion",
                input_text="I just got engaged and I'm so happy I can barely contain it! I want to share this joy with everyone and I feel like my heart might burst. How do I handle all this positive energy?",
                expected_outcomes={
                    "dominant_sense": "E",
                    "domain": "chemistry",
                    "emotional_score_range": [8.0, 10.0],
                    "expansion_support": True,
                    "celebration_integration": True
                },
                success_criteria={
                    "overall_success": 0.85,
                    "joy_amplification": 0.9,
                    "integration_guidance": 0.8,
                    "wave_completion": 1.0
                },
                tags=["joy", "celebration", "expansion", "positive"],
                difficulty_level=3
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_logical_analysis_tests(self):
        """Logical analysis test scenarios"""
        
        scenarios = [
            TestScenario(
                id="logic_001",
                name="Complex Business Decision",
                category=TestCategory.LOGICAL_ANALYSIS,
                description="Test complex multi-factor business analysis",
                input_text="Our company needs to decide between expanding internationally or investing in R&D. International expansion could increase revenue by 40% but requires $2M investment and carries regulatory risks in 3 countries. R&D investment is $1.5M with potential for breakthrough product in 18 months but uncertain market reception. We have limited cash flow and need ROI within 2 years.",
                expected_outcomes={
                    "dominant_sense": "L",
                    "domain": "physics",
                    "logical_score_range": [8.0, 10.0],
                    "structured_analysis": True,
                    "risk_assessment": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "analytical_depth": 0.9,
                    "decision_framework": 0.8,
                    "practical_recommendations": 0.85
                },
                tags=["business", "decision", "analysis", "complex"],
                difficulty_level=5,
                configuration_requirements=["decision_making_pro"]
            ),
            
            TestScenario(
                id="logic_002",
                name="Technical Problem Solving",
                category=TestCategory.LOGICAL_ANALYSIS,
                description="Test systematic technical problem decomposition",
                input_text="Our software system is experiencing intermittent failures that occur randomly across different user sessions. The failures don't correlate with user load, specific features, or time of day. Error logs show memory spikes before crashes, but memory usage appears normal during stable periods. How should we approach debugging this?",
                expected_outcomes={
                    "dominant_sense": "L",
                    "domain": "physics",
                    "logical_score_range": [7.5, 9.5],
                    "systematic_approach": True,
                    "debugging_methodology": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "problem_decomposition": 0.85,
                    "methodical_approach": 0.9,
                    "actionable_steps": 0.8
                },
                tags=["technical", "debugging", "systematic", "problem-solving"],
                difficulty_level=4
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_holding_space_tests(self):
        """Holding space test scenarios"""
        
        scenarios = [
            TestScenario(
                id="holding_001",
                name="Existential Uncertainty",
                category=TestCategory.HOLDING_SPACE,
                description="Test holding space for existential questioning",
                input_text="I've been questioning everything lately - my purpose, my beliefs, what really matters. I don't need answers right now, I just need to sit with these big questions without feeling like I have to figure it all out immediately.",
                expected_outcomes={
                    "dominant_sense": "H",
                    "domain": "biology",
                    "holding_score_range": [7.0, 9.0],
                    "space_holding_quality": True,
                    "non_forcing_approach": True
                },
                success_criteria={
                    "overall_success": 0.85,
                    "holding_capacity": 0.9,
                    "non_directive_support": 0.85,
                    "emergence_allowance": 0.8
                },
                tags=["existential", "uncertainty", "patience", "non-directive"],
                difficulty_level=4
            ),
            
            TestScenario(
                id="holding_002",
                name="Creative Gestation",
                category=TestCategory.HOLDING_SPACE,
                description="Test holding space for creative process",
                input_text="I have this creative project brewing but it's not ready to emerge yet. I can feel something wanting to be born but I need to trust the process and not force it. How do I stay connected to the creative energy without pushing?",
                expected_outcomes={
                    "dominant_sense": "H",
                    "domain": "biology",
                    "holding_score_range": [8.0, 10.0],
                    "creative_support": True,
                    "organic_timing": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "creative_holding": 0.9,
                    "trust_building": 0.85,
                    "process_support": 0.8
                },
                tags=["creative", "gestation", "trust", "organic"],
                difficulty_level=3,
                configuration_requirements=["creative_flow_enhanced"]
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_integration_quality_tests(self):
        """Integration quality test scenarios"""
        
        scenarios = [
            TestScenario(
                id="integration_001",
                name="Head-Heart Integration",
                category=TestCategory.INTEGRATION_QUALITY,
                description="Test integration of logical and emotional perspectives",
                input_text="My head tells me to take the higher-paying job for financial security, but my heart is drawn to the nonprofit work that feels meaningful but pays less. I need to honor both my practical needs and my values. How do I integrate these different kinds of wisdom?",
                expected_outcomes={
                    "dominant_sense": "I",
                    "domain": "chemistry",
                    "integration_score_range": [7.5, 9.5],
                    "synthesis_quality": True,
                    "both_perspectives_honored": True
                },
                success_criteria={
                    "overall_success": 0.85,
                    "integration_quality": 0.9,
                    "synthesis_coherence": 0.85,
                    "practical_wisdom": 0.8
                },
                tags=["integration", "head-heart", "values", "practical"],
                difficulty_level=4
            ),
            
            TestScenario(
                id="integration_002",
                name="Multi-Perspective Synthesis",
                category=TestCategory.INTEGRATION_QUALITY,
                description="Test synthesis of multiple conflicting perspectives",
                input_text="I'm torn between three different life paths: continuing my corporate career for stability, going back to school for a career change, or starting my own business. Each path has compelling arguments and significant risks. I need to find a way to honor the wisdom in each option while making a coherent choice.",
                expected_outcomes={
                    "dominant_sense": "I",
                    "domain": "physics",
                    "integration_score_range": [8.0, 10.0],
                    "multiple_perspective_synthesis": True,
                    "coherent_resolution": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "multi_perspective_integration": 0.9,
                    "decision_coherence": 0.85,
                    "risk_integration": 0.8
                },
                tags=["multi-perspective", "synthesis", "complex-decision", "integration"],
                difficulty_level=5
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_wave_progression_tests(self):
        """Wave progression test scenarios"""
        
        scenarios = [
            TestScenario(
                id="wave_001",
                name="Full Seven-Stage Progression",
                category=TestCategory.WAVE_PROGRESSION,
                description="Test complete wave progression through all seven stages",
                input_text="I'm at a major life transition - leaving a 20-year career to pursue something completely different. I'm excited but also terrified. I need guidance on how to navigate this transformation mindfully and trust the process.",
                expected_outcomes={
                    "wave_stages_completed": 7,
                    "transcendence_achieved": True,
                    "stage_progression_quality": 0.9,
                    "momentum_maintenance": True
                },
                success_criteria={
                    "overall_success": 0.85,
                    "full_wave_completion": 1.0,
                    "stage_coherence": 0.9,
                    "transcendent_insight": 0.8
                },
                tags=["full-wave", "transformation", "transcendence", "complete"],
                difficulty_level=4
            ),
            
            TestScenario(
                id="wave_002",
                name="Wave Disruption Recovery",
                category=TestCategory.WAVE_PROGRESSION,
                description="Test recovery from wave disruption",
                input_text="I keep getting confused and contradicting myself. One minute I think I should do X, then I convince myself Y is better, then I'm back to X again. I feel stuck in loops.",
                expected_outcomes={
                    "disruption_detection": True,
                    "recovery_mechanism": True,
                    "loop_breaking": True,
                    "stabilization_achieved": True
                },
                success_criteria={
                    "overall_success": 0.7,
                    "disruption_recovery": 0.8,
                    "loop_resolution": 0.75,
                    "stabilization_quality": 0.8
                },
                tags=["disruption", "loops", "recovery", "stabilization"],
                difficulty_level=4
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_domain_selection_tests(self):
        """Domain selection test scenarios"""
        
        scenarios = [
            TestScenario(
                id="domain_001",
                name="Chemistry Domain Selection",
                category=TestCategory.DOMAIN_SELECTION,
                description="Test scenarios that should trigger chemistry domain",
                input_text="I feel like I'm going through a complete transformation - old patterns are dissolving and new ones are forming. It's like a chemical reaction happening inside me where the elements of my life are combining in new ways.",
                expected_outcomes={
                    "domain": "chemistry",
                    "transformation_metaphors": True,
                    "reaction_language": True,
                    "catalytic_insights": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "domain_accuracy": 1.0,
                    "metaphor_coherence": 0.85,
                    "transformation_support": 0.8
                },
                tags=["chemistry", "transformation", "domain-selection", "catalytic"],
                difficulty_level=3
            ),
            
            TestScenario(
                id="domain_002",
                name="Biology Domain Selection",
                category=TestCategory.DOMAIN_SELECTION,
                description="Test scenarios that should trigger biology domain",
                input_text="I'm learning to trust my body's wisdom and natural rhythms. Like nature, I go through seasons of growth and rest, expansion and contraction. I want to align with these organic cycles rather than forcing artificial timelines.",
                expected_outcomes={
                    "domain": "biology",
                    "organic_metaphors": True,
                    "natural_rhythm_recognition": True,
                    "adaptive_guidance": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "domain_accuracy": 1.0,
                    "organic_wisdom": 0.85,
                    "rhythm_support": 0.8
                },
                tags=["biology", "organic", "rhythms", "domain-selection"],
                difficulty_level=3
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_edge_case_tests(self):
        """Edge case test scenarios"""
        
        scenarios = [
            TestScenario(
                id="edge_001",
                name="Extremely Short Input",
                category=TestCategory.EDGE_CASES,
                description="Test handling of very brief input",
                input_text="Help.",
                expected_outcomes={
                    "graceful_handling": True,
                    "clarification_request": True,
                    "minimal_viable_response": True
                },
                success_criteria={
                    "overall_success": 0.6,
                    "error_avoidance": 1.0,
                    "helpful_response": 0.7
                },
                tags=["edge-case", "minimal-input", "brevity"],
                difficulty_level=3
            ),
            
            TestScenario(
                id="edge_002",
                name="Contradictory Input",
                category=TestCategory.EDGE_CASES,
                description="Test handling of self-contradictory input",
                input_text="I love my job but I hate it. I want to quit but I never want to leave. I'm happy but miserable. Everything is perfect and terrible at the same time. Nothing makes sense.",
                expected_outcomes={
                    "contradiction_recognition": True,
                    "paradox_holding": True,
                    "integration_attempt": True,
                    "coherence_building": True
                },
                success_criteria={
                    "overall_success": 0.7,
                    "paradox_handling": 0.8,
                    "integration_effort": 0.75,
                    "coherence_achievement": 0.7
                },
                tags=["edge-case", "contradiction", "paradox", "complexity"],
                difficulty_level=5
            ),
            
            TestScenario(
                id="edge_003",
                name="Extremely Long Input",
                category=TestCategory.EDGE_CASES,
                description="Test handling of very long, detailed input",
                input_text="I've been thinking about this situation for months and there are so many layers to consider. First, there's my relationship with my partner who has been going through their own struggles with depression and anxiety, which affects our communication patterns and intimacy levels. Then there's my work situation where I'm dealing with a micromanaging boss who doesn't trust my expertise despite my 15 years of experience in this field. This creates daily stress that I bring home, which then impacts my relationship. Additionally, my aging parents need more support, and I'm the only child nearby, so I feel responsible for their care while also trying to maintain my own mental health and boundaries. On top of all this, I'm questioning my career path and wondering if I should make a major change, but the financial implications are scary, especially with all these other responsibilities. I feel pulled in so many directions and don't know how to prioritize or where to start making changes. Sometimes I think about just running away from it all, but I know that's not realistic or helpful. I need a way to approach all of this systematically without getting overwhelmed, but every time I try to think about one aspect, my mind jumps to another concern.",
                expected_outcomes={
                    "complexity_handling": True,
                    "prioritization_support": True,
                    "overwhelm_management": True,
                    "systematic_approach": True
                },
                success_criteria={
                    "overall_success": 0.75,
                    "complexity_navigation": 0.8,
                    "overwhelm_reduction": 0.8,
                    "practical_guidance": 0.75
                },
                tags=["edge-case", "complexity", "overwhelm", "multi-factor"],
                difficulty_level=5,
                estimated_processing_time=15.0
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_performance_tests(self):
        """Performance test scenarios"""
        
        scenarios = [
            TestScenario(
                id="perf_001",
                name="Speed Benchmark",
                category=TestCategory.PERFORMANCE,
                description="Standard performance benchmark test",
                input_text="I need to make a decision about whether to accept a job offer. The new job pays 20% more but requires relocating and longer hours. How should I weigh these factors?",
                expected_outcomes={
                    "processing_time_ms": 3000,
                    "quality_maintenance": True,
                    "resource_efficiency": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "speed_target": 3.0,  # seconds
                    "quality_threshold": 0.75
                },
                tags=["performance", "benchmark", "speed"],
                difficulty_level=3
            ),
            
            TestScenario(
                id="perf_002",
                name="Memory Efficiency",
                category=TestCategory.PERFORMANCE,
                description="Test memory usage optimization",
                input_text="I'm processing a complex family situation with multiple perspectives and historical context that requires holding a lot of information simultaneously while finding patterns and insights.",
                expected_outcomes={
                    "memory_efficiency": True,
                    "context_retention": True,
                    "processing_optimization": True
                },
                success_criteria={
                    "overall_success": 0.75,
                    "memory_usage": 0.8,  # efficiency score
                    "context_accuracy": 0.8
                },
                tags=["performance", "memory", "efficiency"],
                difficulty_level=4
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_therapeutic_tests(self):
        """Therapeutic application test scenarios"""
        
        scenarios = [
            TestScenario(
                id="therapeutic_001",
                name="Depression Support",
                category=TestCategory.THERAPEUTIC,
                description="Test supportive response to depression",
                input_text="I've been feeling empty and hopeless for weeks. Nothing seems to matter anymore and I'm struggling to get through basic daily tasks. I feel like I'm a burden to everyone around me.",
                expected_outcomes={
                    "therapeutic_response": True,
                    "validation_provided": True,
                    "hope_instillation": True,
                    "resource_offering": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "therapeutic_quality": 0.9,
                    "safety_assessment": 1.0,
                    "support_effectiveness": 0.85
                },
                tags=["therapeutic", "depression", "support", "validation"],
                difficulty_level=5,
                configuration_requirements=["therapeutic_support"],
                validation_rules={
                    "crisis_detection": True,
                    "professional_referral": True,
                    "safety_check": True
                }
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_decision_making_tests(self):
        """Decision making test scenarios"""
        
        scenarios = [
            TestScenario(
                id="decision_001",
                name="Career Transition Decision",
                category=TestCategory.DECISION_MAKING,
                description="Complex career transition with multiple factors",
                input_text="I'm 35 and considering leaving my stable engineering job to become a teacher. I'd need to go back to school for certification, take a 40% pay cut, but I'd find the work more meaningful. I have a mortgage and want to start a family soon.",
                expected_outcomes={
                    "decision_framework": True,
                    "factor_analysis": True,
                    "trade_off_evaluation": True,
                    "recommendations": True
                },
                success_criteria={
                    "overall_success": 0.85,
                    "analytical_depth": 0.9,
                    "practical_guidance": 0.85,
                    "decision_clarity": 0.8
                },
                tags=["decision", "career", "transition", "analysis"],
                difficulty_level=4,
                configuration_requirements=["decision_making_pro"]
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_creative_flow_tests(self):
        """Creative flow test scenarios"""
        
        scenarios = [
            TestScenario(
                id="creative_001",
                name="Creative Block Breakthrough",
                category=TestCategory.CREATIVE_FLOW,
                description="Assistance with creative block and inspiration",
                input_text="I'm a writer and I've been staring at a blank page for days. I have ideas but nothing feels right. I'm stuck in perfectionism and self-criticism. How do I get back into creative flow?",
                expected_outcomes={
                    "creative_support": True,
                    "block_dissolution": True,
                    "flow_facilitation": True,
                    "inspiration_catalysis": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "creative_breakthrough": 0.85,
                    "flow_induction": 0.8,
                    "inspiration_quality": 0.75
                },
                tags=["creative", "block", "flow", "inspiration"],
                difficulty_level=4,
                configuration_requirements=["creative_flow_enhanced"]
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_conflict_resolution_tests(self):
        """Conflict resolution test scenarios"""
        
        scenarios = [
            TestScenario(
                id="conflict_001",
                name="Interpersonal Conflict Mediation",
                category=TestCategory.CONFLICT_RESOLUTION,
                description="Support for resolving interpersonal conflicts",
                input_text="My business partner and I are in constant disagreement about company direction. They want to grow aggressively while I prefer sustainable growth. Our different approaches are creating tension and affecting team morale.",
                expected_outcomes={
                    "perspective_mapping": True,
                    "common_ground_identification": True,
                    "solution_synthesis": True,
                    "relationship_repair": True
                },
                success_criteria={
                    "overall_success": 0.8,
                    "conflict_resolution": 0.85,
                    "relationship_improvement": 0.8,
                    "practical_solutions": 0.8
                },
                tags=["conflict", "mediation", "partnership", "resolution"],
                difficulty_level=4,
                configuration_requirements=["conflict_resolution_mediation"]
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def add_crisis_intervention_tests(self):
        """Crisis intervention test scenarios"""
        
        scenarios = [
            TestScenario(
                id="crisis_001",
                name="Acute Stress Response",
                category=TestCategory.CRISIS_INTERVENTION,
                description="Immediate support for acute stress situation",
                input_text="I just got fired unexpectedly and I'm panicking. I don't know how I'll pay my bills and I feel like my world is falling apart. I can't think straight and my heart is racing.",
                expected_outcomes={
                    "immediate_stabilization": True,
                    "grounding_techniques": True,
                    "practical_support": True,
                    "crisis_management": True
                },
                success_criteria={
                    "overall_success": 0.85,
                    "stabilization_effectiveness": 0.9,
                    "practical_guidance": 0.8,
                    "safety_maintenance": 1.0
                },
                tags=["crisis", "acute-stress", "stabilization", "emergency"],
                difficulty_level=5,
                configuration_requirements=["crisis_intervention"],
                validation_rules={
                    "immediate_response": True,
                    "stabilization_priority": True,
                    "resource_mobilization": True
                }
            )
        ]
        
        for scenario in scenarios:
            self.scenarios[scenario.id] = scenario
    
    def get_scenarios_by_category(self, category: TestCategory) -> List[TestScenario]:
        """Get all scenarios in a specific category"""
        return [scenario for scenario in self.scenarios.values() 
                if scenario.category == category]
    
    def get_scenarios_by_difficulty(self, difficulty: int) -> List[TestScenario]:
        """Get scenarios by difficulty level"""
        return [scenario for scenario in self.scenarios.values()
                if scenario.difficulty_level == difficulty]
    
    def get_scenarios_by_tags(self, tags: List[str]) -> List[TestScenario]:
        """Get scenarios containing any of the specified tags"""
        return [scenario for scenario in self.scenarios.values()
                if any(tag in scenario.tags for tag in tags)]
    
    def create_test_suite(self, 
                         categories: Optional[List[TestCategory]] = None,
                         difficulty_range: Optional[Tuple[int, int]] = None,
                         tags: Optional[List[str]] = None,
                         max_scenarios: int = 50) -> List[TestScenario]:
        """Create a custom test suite based on criteria"""
        
        scenarios = list(self.scenarios.values())
        
        # Filter by categories
        if categories:
            scenarios = [s for s in scenarios if s.category in categories]
        
        # Filter by difficulty
        if difficulty_range:
            min_diff, max_diff = difficulty_range
            scenarios = [s for s in scenarios 
                        if min_diff <= s.difficulty_level <= max_diff]
        
        # Filter by tags
        if tags:
            scenarios = [s for s in scenarios 
                        if any(tag in s.tags for tag in tags)]
        
        # Limit number of scenarios
        scenarios = scenarios[:max_scenarios]
        
        return scenarios
    
    def export_scenarios(self, filename: str, scenarios: Optional[List[TestScenario]] = None):
        """Export scenarios to JSON file"""
        
        if scenarios is None:
            scenarios = list(self.scenarios.values())
        
        export_data = {
            "export_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_scenarios": len(scenarios),
                "categories": list(set(s.category.value for s in scenarios))
            },
            "scenarios": [asdict(scenario) for scenario in scenarios]
        }
        
        # Convert datetime objects to strings
        for scenario_data in export_data["scenarios"]:
            scenario_data["created_at"] = scenario_data["created_at"].isoformat()
            scenario_data["category"] = scenario_data["category"].value
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
    
    def import_scenarios(self, filename: str):
        """Import scenarios from JSON file"""
        
        with open(filename, 'r') as f:
            import_data = json.load(f)
        
        for scenario_data in import_data["scenarios"]:
            # Convert strings back to objects
            scenario_data["created_at"] = datetime.fromisoformat(scenario_data["created_at"])
            scenario_data["category"] = TestCategory(scenario_data["category"])
            
            scenario = TestScenario(**scenario_data)
            self.scenarios[scenario.id] = scenario
    
    def generate_performance_report(self, test_results: List[Dict]) -> Dict:
        """Generate performance analysis report from test results"""
        
        if not test_results:
            return {"error": "No test results provided"}
        
        # Analyze results by category
        category_performance = {}
        for result in test_results:
            category = result.get("category", "unknown")
            if category not in category_performance:
                category_performance[category] = []
            category_performance[category].append(result.get("success_score", 0))
        
        # Calculate averages
        category_averages = {
            category: sum(scores) / len(scores)
            for category, scores in category_performance.items()
        }
        
        # Overall statistics
        all_scores = [result.get("success_score", 0) for result in test_results]
        overall_average = sum(all_scores) / len(all_scores) if all_scores else 0
        
        # Performance analysis
        performance_analysis = {
            "overall_performance": {
                "average_success_rate": overall_average,
                "total_tests": len(test_results),
                "passed_tests": len([r for r in test_results if r.get("success_score", 0) >= 0.7]),
                "failed_tests": len([r for r in test_results if r.get("success_score", 0) < 0.7])
            },
            "category_performance": category_averages,
            "recommendations": self.generate_recommendations(category_averages),
            "timestamp": datetime.now().isoformat()
        }
        
        return performance_analysis
    
    def generate_recommendations(self, category_averages: Dict) -> List[str]:
        """Generate improvement recommendations based on performance"""
        
        recommendations = []
        
        for category, average in category_averages.items():
            if average < 0.6:
                recommendations.append(f"Critical improvement needed in {category} (score: {average:.2f})")
            elif average < 0.7:
                recommendations.append(f"Improvement needed in {category} (score: {average:.2f})")
            elif average < 0.8:
                recommendations.append(f"Minor optimization possible in {category} (score: {average:.2f})")
            else:
                recommendations.append(f"Excellent performance in {category} (score: {average:.2f})")
        
        return recommendations

def main():
    """Demo usage of the test scenario library"""
    
    # Initialize library
    library = TPSTestScenarioLibrary()
    
    print(f"TPS Test Scenario Library initialized with {len(library.scenarios)} scenarios")
    print(f"Categories available: {len(set(s.category for s in library.scenarios.values()))}")
    
    # Create a basic test suite
    basic_suite = library.create_test_suite(
        categories=[TestCategory.BASIC_FUNCTIONALITY, TestCategory.EMOTIONAL_PROCESSING],
        difficulty_range=(1, 3),
        max_scenarios=10
    )
    
    print(f"\nBasic test suite created with {len(basic_suite)} scenarios:")
    for scenario in basic_suite:
        print(f"  - {scenario.name} (difficulty: {scenario.difficulty_level})")
    
    # Export scenarios
    library.export_scenarios("tps_test_scenarios.json")
    print(f"\nScenarios exported to tps_test_scenarios.json")

if __name__ == "__main__":
    main()
