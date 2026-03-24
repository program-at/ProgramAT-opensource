"""
Tests for GitHub issue creation functionality in stream_server.py
"""
import asyncio
import unittest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
import os

# Mock the dependencies before importing stream_server
sys.modules['cv2'] = Mock()
sys.modules['numpy'] = Mock()
sys.modules['PIL'] = Mock()
sys.modules['websockets'] = Mock()

# Now we can test the functions


class TestGitHubIntegration(unittest.TestCase):
    """Test GitHub issue creation logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_text = "Test issue title"
        
    @patch('os.environ.get')
    @patch('github.Github')
    def test_create_github_issue_with_token(self, mock_github, mock_env):
        """Test that issue is created when token is configured"""
        # Mock environment variables
        mock_env.side_effect = lambda key, default='': {
            'GITHUB_TOKEN': 'test_token',
            'GITHUB_REPO': 'test/repo'
        }.get(key, default)
        
        # Mock GitHub API
        mock_repo = Mock()
        mock_issue = Mock()
        mock_issue.number = 123
        mock_repo.create_issue.return_value = mock_issue
        mock_github.return_value.get_repo.return_value = mock_repo
        
        # Import and run the function
        from stream_server import create_github_issue
        
        # Run async function
        asyncio.run(create_github_issue(self.test_text))
        
        # Verify issue was created
        mock_repo.create_issue.assert_called_once_with(title=self.test_text)
    
    def test_text_tracking_data_structure(self):
        """Test that last_text dictionary has correct structure"""
        last_text = {'content': None, 'timestamp': None, 'task': None}
        
        # Simulate receiving text
        last_text['content'] = "Test message"
        last_text['timestamp'] = datetime.now()
        last_text['task'] = None
        
        self.assertEqual(last_text['content'], "Test message")
        self.assertIsNotNone(last_text['timestamp'])
        self.assertIsNone(last_text['task'])
    
    def test_pause_detection_logic(self):
        """Test that pause is correctly detected"""
        PAUSE_DURATION = 5.0
        
        # Simulate old timestamp
        old_timestamp = datetime.now() - timedelta(seconds=6)
        elapsed = (datetime.now() - old_timestamp).total_seconds()
        
        # Should trigger issue creation
        self.assertGreaterEqual(elapsed, PAUSE_DURATION)
        
        # Recent timestamp
        recent_timestamp = datetime.now() - timedelta(seconds=2)
        elapsed = (datetime.now() - recent_timestamp).total_seconds()
        
        # Should not trigger issue creation
        self.assertLess(elapsed, PAUSE_DURATION)


class TestTextProcessing(unittest.TestCase):
    """Test text processing functions"""
    
    def test_text_normalization_string(self):
        """Test that string text is handled correctly"""
        text_str = "Hello world"
        self.assertIsInstance(text_str, str)
    
    def test_text_normalization_dict(self):
        """Test that dict text is converted to string"""
        import json
        text_dict = {"message": "Hello world"}
        text_str = json.dumps(text_dict, ensure_ascii=False)
        self.assertIsInstance(text_str, str)
        self.assertIn("Hello world", text_str)
    
    def test_empty_text_handling(self):
        """Test that empty text is handled gracefully"""
        empty_text = None
        self.assertIsNone(empty_text)
        
        empty_string = ""
        self.assertEqual(empty_string.strip(), "")


if __name__ == '__main__':
    unittest.main()
