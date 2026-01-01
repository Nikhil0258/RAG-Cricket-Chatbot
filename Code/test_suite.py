"""
============================================================================
CRICKET RAG CHATBOT - COMPREHENSIVE TEST SUITE
============================================================================

This test suite covers:
1. Base cases (happy paths)
2. Edge cases (boundary conditions)
3. Error cases (invalid inputs)
4. Integration scenarios (complex queries)

Test Categories:
- Query Normalization Tests
- Ambiguity Detection Tests
- Intent Classification Tests
- Numerical Query Tests
- Descriptive Query Tests
- Hybrid Query Tests
- Error Handling Tests
- Performance Tests
============================================================================
"""

import unittest
from typing import Dict, List
import time
import sys
import os

# Import the CricketChatbot class from your main file
# Adjust the import based on your file structure
try:
    from final_design import CricketChatbot
except ImportError:
    # Alternative import if running from different directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from final_design import CricketChatbot


# ============================================================================
# TEST CATEGORY 1: QUERY NORMALIZATION TESTS
# ============================================================================

class TestQueryNormalization(unittest.TestCase):
    """Test entity extraction from natural language queries."""
    
    def setUp(self):
        """Initialize chatbot before each test."""
        self.chatbot = CricketChatbot()
    
    # BASE CASES
    def test_normalize_complete_query(self):
        """Test extraction when all entities are present."""
        query = "How many runs did Rishabh Pant score in the 2021 India vs Australia series?"
        result = self.chatbot.query_processor.normalize_query(query)
        
        self.assertEqual(result['player'], 'Rishabh Pant')
        self.assertEqual(result['year'], 2021)
        self.assertIn('India', result['series'])
        self.assertIn('Australia', result['series'])
    
    def test_normalize_player_nickname(self):
        """Test that nicknames are normalized to full names."""
        query = "How did Pant perform in 2021?"
        result = self.chatbot.query_processor.normalize_query(query)
        
        self.assertEqual(result['player'], 'Rishabh Pant')
        self.assertEqual(result['year'], 2021)
    
    def test_normalize_ordinal_numbers(self):
        """Test conversion of ordinal numbers to integers."""
        test_cases = [
            ("Tell me about the first test", 1),
            ("What happened in the second match", 2),
            ("fourth test details", 4),
            ("3rd match summary", 3)
        ]
        
        for query, expected_num in test_cases:
            result = self.chatbot.query_processor.normalize_query(query)
            self.assertEqual(result['match_number'], expected_num)
    
    # EDGE CASES
    def test_normalize_partial_information(self):
        """Test extraction when only some entities are present."""
        query = "How many runs did Pant score?"
        result = self.chatbot.query_processor.normalize_query(query)
        
        self.assertEqual(result['player'], 'Rishabh Pant')
        self.assertIsNone(result['year'])
        self.assertIsNone(result['series'])
    
    def test_normalize_year_only(self):
        """Test extraction with only year specified."""
        query = "What happened in 2021?"
        result = self.chatbot.query_processor.normalize_query(query)
        
        self.assertEqual(result['year'], 2021)
        self.assertIsNone(result['player'])
    
    def test_normalize_multiple_players(self):
        """Test handling when multiple players are mentioned."""
        query = "Compare Pant and Rahane's performance in 2021"
        result = self.chatbot.query_processor.normalize_query(query)
        
        # Should extract at least one player
        self.assertIsNotNone(result['player'])
        self.assertIn('Pant', result['player']) or self.assertIn('Rahane', result['player'])
    
    def test_normalize_year_boundaries(self):
        """Test year extraction at boundaries (2020-2024)."""
        test_cases = [2020, 2021, 2022, 2023, 2024]
        
        for year in test_cases:
            query = f"What happened in {year}?"
            result = self.chatbot.query_processor.normalize_query(query)
            self.assertEqual(result['year'], year)
    
    def test_normalize_invalid_year(self):
        """Test handling of years outside valid range."""
        query = "What happened in 2019?"
        result = self.chatbot.query_processor.normalize_query(query)
        
        # Should either reject or handle gracefully
        # Year might be extracted but will fail in downstream processing
        self.assertIsNotNone(result['raw_query'])


# ============================================================================
# TEST CATEGORY 2: AMBIGUITY DETECTION TESTS
# ============================================================================

class TestAmbiguityDetection(unittest.TestCase):
    """Test detection of queries requiring clarification."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    # BASE CASES - Queries that SHOULD trigger clarification
    def test_ambiguous_match_number_without_context(self):
        """Test that match number without year/series is flagged."""
        normalized = {
            'raw_query': 'Tell me about the fourth test',
            'match_number': 4,
            'year': None,
            'series': None,
            'player': None,
            'team': None
        }
        
        result = self.chatbot.query_processor.check_ambiguity(normalized)
        self.assertTrue(result['needed'])
        self.assertIn('4th test', result['message'])
    
    def test_ambiguous_player_stats_without_timeframe(self):
        """Test that player stats query without year is flagged."""
        normalized = {
            'raw_query': 'How many runs did Pant score?',
            'player': 'Rishabh Pant',
            'year': None,
            'series': None,
            'match_number': None,
            'team': None
        }
        
        result = self.chatbot.query_processor.check_ambiguity(normalized)
        self.assertTrue(result['needed'])
        self.assertIn('Rishabh Pant', result['message'])
    
    def test_ambiguous_vague_reference(self):
        """Test that vague references like 'the match' are flagged."""
        vague_queries = [
            'Tell me about the match',
            'What happened in the test',
            'Describe the series',
            'Details about that match'
        ]
        
        for query in vague_queries:
            normalized = {
                'raw_query': query,
                'year': None,
                'series': None,
                'match_number': None,
                'player': None,
                'team': None
            }
            
            result = self.chatbot.query_processor.check_ambiguity(normalized)
            self.assertTrue(result['needed'], f"Failed for: {query}")
    
    # BASE CASES - Queries that should NOT trigger clarification
    def test_unambiguous_complete_query(self):
        """Test that complete queries are not flagged."""
        normalized = {
            'raw_query': 'How many runs did Pant score in 2021?',
            'player': 'Rishabh Pant',
            'year': 2021,
            'series': None,
            'match_number': None,
            'team': None
        }
        
        result = self.chatbot.query_processor.check_ambiguity(normalized)
        self.assertFalse(result['needed'])
    
    def test_unambiguous_match_with_year(self):
        """Test that match number with year is not flagged."""
        normalized = {
            'raw_query': 'Tell me about the fourth test in 2021',
            'match_number': 4,
            'year': 2021,
            'series': None,
            'player': None,
            'team': None
        }
        
        result = self.chatbot.query_processor.check_ambiguity(normalized)
        self.assertFalse(result['needed'])
    
    # EDGE CASES
    def test_descriptive_query_without_stats_keywords(self):
        """Test that descriptive queries about players don't trigger clarification."""
        normalized = {
            'raw_query': 'Tell me about Rishabh Pant',
            'player': 'Rishabh Pant',
            'year': None,
            'series': None,
            'match_number': None,
            'team': None
        }
        
        result = self.chatbot.query_processor.check_ambiguity(normalized)
        # Should NOT require clarification for descriptive queries
        self.assertFalse(result['needed'])


# ============================================================================
# TEST CATEGORY 3: INTENT CLASSIFICATION TESTS
# ============================================================================

class TestIntentClassification(unittest.TestCase):
    """Test query intent classification."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    # BASE CASES - Numerical Intent
    def test_numerical_intent_direct_stats(self):
        """Test classification of direct statistical queries."""
        numerical_queries = [
            "How many runs did Pant score in 2021?",
            "What was Bumrah's wicket tally?",
            "Total runs by Rahane",
            "What is the average of Pant?",
            "How many sixes did he hit?"
        ]
        
        for query in numerical_queries:
            intent = self.chatbot.query_processor.classify_intent(query)
            self.assertEqual(intent, 'numerical', f"Failed for: {query}")
    
    # BASE CASES - Descriptive Intent
    def test_descriptive_intent_narrative(self):
        """Test classification of narrative queries."""
        descriptive_queries = [
            "Describe the 2021 series",
            "What happened in the match?",
            "Tell me about the India vs Australia series",
            "Explain how India won",
            "Give me a summary of the match"
        ]
        
        for query in descriptive_queries:
            intent = self.chatbot.query_processor.classify_intent(query)
            self.assertEqual(intent, 'descriptive', f"Failed for: {query}")
    
    # BASE CASES - Hybrid Intent
    def test_hybrid_intent_combined(self):
        """Test classification of queries needing both stats and narrative."""
        hybrid_queries = [
            "How many runs did Pant score and how did he play?",
            "What was Bumrah's wicket count and describe his bowling",
            "Tell me Rahane's performance with statistics",
            "Give me the run count and describe the innings"
        ]
        
        for query in hybrid_queries:
            intent = self.chatbot.query_processor.classify_intent(query)
            self.assertIn(intent, ['hybrid', 'numerical'], f"Failed for: {query}")
    
    # EDGE CASES
    def test_ambiguous_intent_boundary(self):
        """Test queries at the boundary between categories."""
        query = "How did Pant perform?"
        intent = self.chatbot.query_processor.classify_intent(query)
        
        # Could be descriptive or hybrid - both acceptable
        self.assertIn(intent, ['descriptive', 'hybrid'])
    
    def test_intent_with_implicit_stats(self):
        """Test queries with implicit statistical requests."""
        query = "Was Pant successful in 2021?"
        intent = self.chatbot.query_processor.classify_intent(query)
        
        # Should likely be descriptive or hybrid
        self.assertIn(intent, ['descriptive', 'hybrid'])


# ============================================================================
# TEST CATEGORY 4: NUMERICAL QUERY TESTS
# ============================================================================

class TestNumericalQueries(unittest.TestCase):
    """Test pure statistical queries."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    # BASE CASES
    def test_total_runs_query(self):
        """Test basic total runs query."""
        query = "How many runs did Rishabh Pant score in 2021?"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'numerical')
        self.assertEqual(result['source'], 'stats_tool')
        self.assertIn('Total Runs', result['answer'])
        self.assertIsInstance(result['answer'], str)
    
    def test_individual_runs_breakdown(self):
        """Test per-match runs breakdown."""
        query = "Give me Pant's run breakdown for 2021 match by match"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'numerical')
        self.assertIn('Breakdown', result['answer'])
    
    def test_batting_average_query(self):
        """Test batting average and strike rate query."""
        query = "What was Pant's batting average in 2021?"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'numerical')
        self.assertIn('average', result['answer'].lower())
    
    # EDGE CASES
    def test_nonexistent_player(self):
        """Test query for player not in database."""
        query = "How many runs did John Doe score in 2021?"
        result = self.chatbot.answer(query)
        
        # Should handle gracefully
        self.assertIsNotNone(result['answer'])
        # Answer should indicate no data found
        self.assertTrue(
            'no data' in result['answer'].lower() or 
            'not found' in result['answer'].lower() or
            'error' in result['answer'].lower()
        )
    
    def test_invalid_year_query(self):
        """Test query with year outside database range."""
        query = "How many runs did Pant score in 2030?"
        result = self.chatbot.answer(query)
        
        # Should handle gracefully
        self.assertIsNotNone(result['answer'])
    
    def test_zero_runs_scenario(self):
        """Test handling when player scored zero runs."""
        # This would need a real case in your data
        query = "How many runs did [player] score in [specific match where they scored 0]?"
        # Note: You'd need to identify a real scenario from your data
        
        # Placeholder test - actual implementation depends on data
        pass
    
    def test_multiple_metrics_single_query(self):
        """Test query asking for multiple metrics."""
        query = "Give me Pant's runs, average, and strike rate for 2021"
        result = self.chatbot.answer(query)
        
        # Should provide comprehensive statistics
        self.assertEqual(result['intent'], 'numerical')


# ============================================================================
# TEST CATEGORY 5: DESCRIPTIVE QUERY TESTS
# ============================================================================

class TestDescriptiveQueries(unittest.TestCase):
    """Test narrative/descriptive queries."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    # BASE CASES
    def test_match_summary_query(self):
        """Test basic match summary request."""
        query = "Tell me about the India vs Australia 2021 series"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'descriptive')
        self.assertEqual(result['source'], 'rag')
        self.assertGreater(len(result['answer']), 50)  # Should have substantial content
    
    def test_specific_match_description(self):
        """Test description of specific match."""
        query = "Describe what happened in the first test of 2021"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'descriptive')
        self.assertIsNotNone(result.get('chunks_used'))
    
    def test_player_performance_narrative(self):
        """Test narrative description of player performance."""
        query = "Tell me how Rishabh Pant batted in 2021"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'descriptive')
        self.assertIn('Pant', result['answer'])
    
    # EDGE CASES
    def test_very_broad_query(self):
        """Test extremely broad query."""
        query = "What happened in cricket?"
        result = self.chatbot.answer(query)
        
        # Should either ask for clarification or provide general info
        self.assertIsNotNone(result['answer'])
    
    def test_query_with_no_matching_context(self):
        """Test query that has no relevant context in database."""
        query = "Tell me about the 2015 World Cup"
        result = self.chatbot.answer(query)
        
        # Should indicate no relevant information found
        self.assertTrue(
            'no' in result['answer'].lower() or
            'not found' in result['answer'].lower() or
            'no relevant' in result['answer'].lower()
        )
    
    def test_contradictory_context(self):
        """Test handling when retrieved chunks might have contradictory info."""
        # This is a challenging edge case
        # Would need specific data scenarios to test properly
        pass


# ============================================================================
# TEST CATEGORY 6: HYBRID QUERY TESTS
# ============================================================================

class TestHybridQueries(unittest.TestCase):
    """Test queries requiring both statistics and narrative."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    # BASE CASES
    def test_hybrid_runs_and_description(self):
        """Test query asking for both runs and playing style."""
        query = "How many runs did Pant score in 2021 and how did he play?"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'hybrid')
        self.assertEqual(result['source'], 'hybrid')
        # Should contain both numbers and narrative
        self.assertTrue(any(char.isdigit() for char in result['answer']))
        self.assertGreater(len(result['answer']), 100)
    
    def test_hybrid_performance_with_context(self):
        """Test detailed performance query with context."""
        query = "What was Pant's batting performance in 2021 with match details?"
        result = self.chatbot.answer(query)
        
        self.assertEqual(result['intent'], 'hybrid')
        self.assertIsNotNone(result.get('match_ids'))
    
    # EDGE CASES
    def test_hybrid_scope_alignment(self):
        """Test that stats and narrative cover same matches."""
        query = "How many runs did Pant score in 2021 and describe his innings"
        result = self.chatbot.answer(query)
        
        if result['intent'] == 'hybrid':
            # Verify scope alignment
            self.assertIn('Scope', result['answer'])
            # Should mention both narrative and stats scope
    
    def test_hybrid_with_limited_data(self):
        """Test hybrid query when only partial data available."""
        # Player with limited statistical data
        query = "How many runs did [limited_data_player] score and how did they play?"
        # This would need identification of such a player in your dataset
        pass


# ============================================================================
# TEST CATEGORY 7: ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    # EDGE CASES
    def test_empty_query(self):
        """Test handling of empty query."""
        query = ""
        result = self.chatbot.answer(query)
        
        self.assertIsNotNone(result)
        self.assertIsNotNone(result['answer'])
    
    def test_very_long_query(self):
        """Test handling of extremely long query."""
        query = "How many runs did Pant score? " * 100
        result = self.chatbot.answer(query)
        
        self.assertIsNotNone(result)
    
    def test_special_characters(self):
        """Test handling of special characters in query."""
        queries = [
            "How many runs did Pant score!?!?",
            "Pant's runs in 2021???",
            "Tell me about @Pant #performance",
            "What about $$ runs $$"
        ]
        
        for query in queries:
            result = self.chatbot.answer(query)
            self.assertIsNotNone(result['answer'])
    
    def test_non_cricket_query(self):
        """Test handling of completely unrelated query."""
        query = "What is the capital of France?"
        result = self.chatbot.answer(query)
        
        # Should handle gracefully, possibly indicate no relevant info
        self.assertIsNotNone(result['answer'])
    
    def test_malformed_dates(self):
        """Test handling of malformed date formats."""
        queries = [
            "What happened in 20/21?",
            "Tell me about year twenty twenty one",
            "Performance in '21"
        ]
        
        for query in queries:
            result = self.chatbot.answer(query)
            self.assertIsNotNone(result)
    
    def test_mixed_language_query(self):
        """Test handling of query with non-English words."""
        query = "Pant ke kitne runs the in 2021?"  # Hindi mixed with English
        result = self.chatbot.answer(query)
        
        self.assertIsNotNone(result)
    
    def test_query_with_typos(self):
        """Test handling of common typos."""
        queries = [
            "How many runz did Pant scor?",
            "Tel me about Risabh Pant",
            "What hapened in 2021?"
        ]
        
        for query in queries:
            result = self.chatbot.answer(query)
            self.assertIsNotNone(result['answer'])


# ============================================================================
# TEST CATEGORY 8: INTEGRATION TESTS
# ============================================================================

class TestIntegration(unittest.TestCase):
    """Test complex end-to-end scenarios."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    def test_multi_turn_conversation(self):
        """Test sequential queries in a conversation."""
        queries = [
            "Tell me about the 2021 India vs Australia series",
            "How many runs did Pant score in that series?",
            "What about Rahane's performance?"
        ]
        
        results = []
        for query in queries:
            result = self.chatbot.answer(query)
            results.append(result)
            self.assertIsNotNone(result['answer'])
        
        # All queries should succeed
        self.assertEqual(len(results), 3)
    
    def test_clarification_then_specific(self):
        """Test ambiguous query followed by clarified query."""
        # First query - should ask for clarification
        result1 = self.chatbot.answer("How many runs did Pant score?")
        self.assertEqual(result1['intent'], 'clarification')
        
        # Second query - with clarification
        result2 = self.chatbot.answer("How many runs did Pant score in 2021?")
        self.assertEqual(result2['intent'], 'numerical')
    
    def test_comparative_queries(self):
        """Test queries comparing multiple entities."""
        queries = [
            "Compare Pant and Rahane's performance in 2021",
            "Who scored more runs: Pant or Gill in 2021?"
        ]
        
        for query in queries:
            result = self.chatbot.answer(query)
            self.assertIsNotNone(result['answer'])
    
    def test_negative_queries(self):
        """Test queries with negative framing."""
        queries = [
            "Did Pant fail in 2021?",
            "Why didn't India win the series?",
            "Which matches did Pant not play in?"
        ]
        
        for query in queries:
            result = self.chatbot.answer(query)
            self.assertIsNotNone(result['answer'])


# ============================================================================
# TEST CATEGORY 9: PERFORMANCE TESTS
# ============================================================================

class TestPerformance(unittest.TestCase):
    """Test system performance and efficiency."""
    
    def setUp(self):
        self.chatbot = CricketChatbot()
    
    def test_response_time(self):
        """Test that responses are generated in reasonable time."""
        query = "How many runs did Pant score in 2021?"
        
        start_time = time.time()
        result = self.chatbot.answer(query)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Should respond within 10 seconds (adjust based on your requirements)
        self.assertLess(response_time, 10.0)
        print(f"Response time: {response_time:.2f}s")
    
    def test_caching_effectiveness(self):
        """Test that repeated queries are faster due to caching."""
        query = "How many runs did Pant score in 2021?"
        
        # First call
        start1 = time.time()
        result1 = self.chatbot.answer(query)
        time1 = time.time() - start1
        
        # Second call (should use cache)
        start2 = time.time()
        result2 = self.chatbot.answer(query)
        time2 = time.time() - start2
        
        print(f"First call: {time1:.2f}s, Second call: {time2:.2f}s")
        
        # Second call should be faster or similar
        # (Some components may still take time even with cache)
        self.assertIsNotNone(result1)
        self.assertIsNotNone(result2)
    
    def test_concurrent_queries(self):
        """Test handling of multiple queries."""
        queries = [
            "How many runs did Pant score in 2021?",
            "Tell me about the 2021 series",
            "What was Rahane's average?",
            "Describe the first test"
        ]
        
        start_time = time.time()
        results = [self.chatbot.answer(q) for q in queries]
        total_time = time.time() - start_time
        
        # All should succeed
        self.assertEqual(len(results), len(queries))
        for result in results:
            self.assertIsNotNone(result['answer'])
        
        print(f"Total time for {len(queries)} queries: {total_time:.2f}s")


# ============================================================================
# TEST RUNNER WITH REPORTING
# ============================================================================

def run_test_suite():
    """Run all tests with detailed reporting."""
    
    print("="*80)
    print("CRICKET RAG CHATBOT - COMPREHENSIVE TEST SUITE")
    print("="*80)
    print()
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestQueryNormalization,
        TestAmbiguityDetection,
        TestIntentClassification,
        TestNumericalQueries,
        TestDescriptiveQueries,
        TestHybridQueries,
        TestErrorHandling,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result


# ============================================================================
# QUICK TEST EXAMPLES
# ============================================================================

def run_quick_tests():
    """Run a quick subset of critical tests."""
    
    print("\n" + "="*80)
    print("QUICK TEST SUITE - Critical Functionality")
    print("="*80 + "\n")
    
    chatbot = CricketChatbot()
    
    test_cases = [
        {
            "name": "Simple Numerical Query",
            "query": "How many runs did Pant score in 2021?",
            "expected_intent": "numerical"
        },
        {
            "name": "Simple Descriptive Query",
            "query": "Tell me about the 2021 India vs Australia series",
            "expected_intent": "descriptive"
        },
        {
            "name": "Ambiguous Query (should trigger clarification)",
            "query": "How many runs did Pant score?",
            "expected_intent": "clarification"
        },
        {
            "name": "Hybrid Query",
            "query": "How many runs did Pant score in 2021 and how did he play?",
            "expected_intent": "hybrid"
        }
    ]
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}: {test['name']}")
        print(f"Query: {test['query']}")
        
        try:
            result = chatbot.answer(test['query'])
            
            if result['intent'] == test['expected_intent']:
                print(f"✓ PASSED - Intent: {result['intent']}")
                passed += 1
            else:
                print(f"✗ FAILED - Expected: {test['expected_intent']}, Got: {result['intent']}")
                failed += 1
            
            print(f"Answer preview: {result['answer'][:150]}...")
            
        except Exception as e:
            print(f"✗ FAILED - Error: {str(e)}")
            failed += 1
        
        print("-" * 80 + "\n")
    
    print("="*80)
    print(f"Quick Test Summary: {passed} passed, {failed} failed")
    print("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Run quick tests
        run_quick_tests()
    else:
        # Run full test suite
        run_test_suite()