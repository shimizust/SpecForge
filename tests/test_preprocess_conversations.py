import pytest
import torch
from unittest.mock import Mock, MagicMock

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


import pytest
import torch
import os
from unittest.mock import Mock, patch

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY


def test_preprocess_conversations_basic():
    """Test basic conversation preprocessing without requiring external models."""
    # Mock the tokenizer to avoid network calls
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
        # Create a mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.unk_token_id = 2
        
        # Mock the apply_chat_template method
        def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            conversation = ""
            for msg in messages:
                if msg["role"] == "system":
                    conversation += f"<|system|>\n{msg['content']}\n<|end|>"
                elif msg["role"] == "user":
                    conversation += f"<|user|>\n{msg['content']}\n<|end|>"
                elif msg["role"] == "assistant":
                    conversation += f"<|assistant|>\n{msg['content']}\n<|end|>"
            return conversation
        
        mock_tokenizer.apply_chat_template = mock_apply_chat_template
        
        # Mock the tokenizer call method
        def mock_tokenize(text, return_offsets_mapping=False, max_length=None, 
                         truncation=False, return_tensors=None, add_special_tokens=True):
            # Simple tokenization based on character positions
            tokens = text.split()
            if max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            token_ids = [i + 3 for i in range(len(tokens))]
            
            # Create a mock encoding object with attributes
            class MockEncoding:
                def __init__(self, input_ids, offset_mapping=None):
                    self.input_ids = input_ids
                    self.offset_mapping = offset_mapping
            
            input_ids = torch.tensor([token_ids]) if return_tensors == "pt" else [token_ids]
            
            if return_offsets_mapping:
                # Create realistic offsets based on the original text
                offsets = []
                char_pos = 0
                for token in tokens:
                    start = text.find(token, char_pos)
                    if start == -1:
                        start = char_pos
                    end = start + len(token)
                    offsets.append((start, end))
                    char_pos = end
                offset_mapping = torch.tensor([offsets]) if return_tensors == "pt" else [offsets]
                return MockEncoding(input_ids, offset_mapping)
            else:
                return MockEncoding(input_ids)
        
        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer_class.return_value = mock_tokenizer
        
        # Test data
        conversations = [
            [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"},
            ]
        ]
        
        chat_template = TEMPLATE_REGISTRY.get("llama3")
        
        # Run the function
        result = preprocess_conversations(
            tokenizer=mock_tokenizer,
            conversations=conversations,
            chat_template=chat_template,
            max_length=512,
        )
        
        # Assertions
        assert "input_ids" in result
        assert "loss_mask" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 1
        assert len(result["loss_mask"]) == 1
        assert len(result["attention_mask"]) == 1
        
        # Check tensor shapes
        input_ids = result["input_ids"][0]
        loss_mask = result["loss_mask"][0]
        attention_mask = result["attention_mask"][0]
        
        assert input_ids.shape == loss_mask.shape
        assert input_ids.shape == attention_mask.shape
        assert len(input_ids.shape) == 2  # Should have batch dimension


def test_preprocess_conversations_multiple():
    """Test preprocessing multiple conversations."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            conversation = ""
            for msg in messages:
                if msg["role"] == "system":
                    conversation += f"<|system|>\n{msg['content']}\n<|end|>"
                elif msg["role"] == "user":
                    conversation += f"<|user|>\n{msg['content']}\n<|end|>"
                elif msg["role"] == "assistant":
                    conversation += f"<|assistant|>\n{msg['content']}\n<|end|>"
            return conversation
        
        mock_tokenizer.apply_chat_template = mock_apply_chat_template
        
        def mock_tokenize(text, return_offsets_mapping=False, max_length=None, 
                         truncation=False, return_tensors=None, add_special_tokens=True):
            tokens = text.split()
            if max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            token_ids = [i + 3 for i in range(len(tokens))]
            
            class MockEncoding:
                def __init__(self, input_ids, offset_mapping=None):
                    self.input_ids = input_ids
                    self.offset_mapping = offset_mapping
            
            input_ids = torch.tensor([token_ids])
            
            if return_offsets_mapping:
                offsets = []
                char_pos = 0
                for token in tokens:
                    start = text.find(token, char_pos)
                    if start == -1:
                        start = char_pos
                    end = start + len(token)
                    offsets.append((start, end))
                    char_pos = end
                offset_mapping = torch.tensor([offsets])
                return MockEncoding(input_ids, offset_mapping)
            else:
                return MockEncoding(input_ids)
        
        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer_class.return_value = mock_tokenizer
        
        conversations = [
            [
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's sunny."},
            ],
            [
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
                {"role": "user", "content": "Why?"},
                {"role": "assistant", "content": "To get to the other side!"},
            ],
        ]
        
        chat_template = TEMPLATE_REGISTRY.get("llama3")
        
        result = preprocess_conversations(
            tokenizer=mock_tokenizer,
            conversations=conversations,
            chat_template=chat_template,
            max_length=512,
        )
        
        # Should process both conversations
        assert len(result["input_ids"]) == 2
        assert len(result["loss_mask"]) == 2
        assert len(result["attention_mask"]) == 2


def test_preprocess_conversations_empty():
    """Test handling of empty conversations."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            conversation = ""
            for msg in messages:
                if msg["role"] == "system":
                    conversation += f"<|system|>\n{msg['content']}\n<|end|>"
                elif msg["role"] == "user":
                    conversation += f"<|user|>\n{msg['content']}\n<|end|>"
                elif msg["role"] == "assistant":
                    conversation += f"<|assistant|>\n{msg['content']}\n<|end|>"
            return conversation
        
        mock_tokenizer.apply_chat_template = mock_apply_chat_template
        
        def mock_tokenize(text, return_offsets_mapping=False, max_length=None, 
                         truncation=False, return_tensors=None, add_special_tokens=True):
            tokens = text.split()
            token_ids = [i + 3 for i in range(len(tokens))]
            
            class MockEncoding:
                def __init__(self, input_ids, offset_mapping=None):
                    self.input_ids = input_ids
                    self.offset_mapping = offset_mapping
            
            input_ids = torch.tensor([token_ids])
            
            if return_offsets_mapping:
                offsets = [(i*5, (i+1)*5) for i in range(len(tokens))]
                offset_mapping = torch.tensor([offsets])
                return MockEncoding(input_ids, offset_mapping)
            else:
                return MockEncoding(input_ids)
        
        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer_class.return_value = mock_tokenizer
        
        conversations = [
            [],  # Empty conversation
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        ]
        
        chat_template = TEMPLATE_REGISTRY.get("llama3")
        
        result = preprocess_conversations(
            tokenizer=mock_tokenizer,
            conversations=conversations,
            chat_template=chat_template,
            max_length=512,
        )
        
        # Empty conversation should be skipped
        assert len(result["input_ids"]) == 1
        assert len(result["loss_mask"]) == 1


def test_preprocess_conversations_role_validation():
    """Test role validation."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        conversations = [
            [
                {"role": "user", "content": "First message"},
                {"role": "user", "content": "Second message"},  # Invalid: two user messages
            ]
        ]
        
        chat_template = TEMPLATE_REGISTRY.get("llama3")
        
        # This should raise an AssertionError
        with pytest.raises(AssertionError, match="unexpected role"):
            preprocess_conversations(
                tokenizer=mock_tokenizer,
                conversations=conversations,
                chat_template=chat_template,
                max_length=512,
            )


def test_preprocess_conversations_none():
    """Test handling of None conversations."""
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0
        
        def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
            return "<|system|>\nYou are a helpful assistant.\n<|end|><|user|>\nHello\n<|end|><|assistant|>\nHi\n<|end|>"
        
        mock_tokenizer.apply_chat_template = mock_apply_chat_template
        
        def mock_tokenize(text, return_offsets_mapping=False, max_length=None, 
                         truncation=False, return_tensors=None, add_special_tokens=True):
            tokens = text.split()
            token_ids = [i + 3 for i in range(len(tokens))]
            
            class MockEncoding:
                def __init__(self, input_ids, offset_mapping=None):
                    self.input_ids = input_ids
                    self.offset_mapping = offset_mapping
            
            input_ids = torch.tensor([token_ids])
            
            if return_offsets_mapping:
                offsets = [(i*5, (i+1)*5) for i in range(len(tokens))]
                offset_mapping = torch.tensor([offsets])
                return MockEncoding(input_ids, offset_mapping)
            else:
                return MockEncoding(input_ids)
        
        mock_tokenizer.side_effect = mock_tokenize
        mock_tokenizer_class.return_value = mock_tokenizer
        
        conversations = [
            None,  # None conversation
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
        ]
        
        chat_template = TEMPLATE_REGISTRY.get("llama3")
        
        result = preprocess_conversations(
            tokenizer=mock_tokenizer,
            conversations=conversations,
            chat_template=chat_template,
            max_length=512,
        )
        
        # None conversation should be skipped
        assert len(result["input_ids"]) == 1


if __name__ == "__main__":
    # Run the tests individually
    print("Running test_preprocess_conversations_basic...")
    test_preprocess_conversations_basic()
    print("✓ Basic test passed")
    
    print("Running test_preprocess_conversations_multiple...")
    test_preprocess_conversations_multiple()
    print("✓ Multiple conversations test passed")
    
    print("Running test_preprocess_conversations_empty...")
    test_preprocess_conversations_empty()
    print("✓ Empty conversations test passed")
    
    print("Running test_preprocess_conversations_role_validation...")
    try:
        test_preprocess_conversations_role_validation()
        print("✗ Role validation test should have raised an error")
    except AssertionError as e:
        if "unexpected role" in str(e):
            print("✓ Role validation test passed")
        else:
            print(f"✗ Role validation test failed with unexpected error: {e}")
    
    print("Running test_preprocess_conversations_none...")
    test_preprocess_conversations_none()
    print("✓ None conversations test passed")
    
    print("\nAll tests completed successfully!")


# Additional utility function for visual debugging (similar to the VLM test)
def visualize_loss_mask(tokenizer, input_ids, loss_mask):
    """Utility function to visualize which tokens contribute to loss."""
    RED = "\033[91m"    # Non-assistant tokens (loss_mask = 0)
    GREEN = "\033[92m"  # Assistant tokens (loss_mask = 1)
    RESET = "\033[0m"
    
    print("\nLoss Mask Visualization:")
    print("RED = Non-assistant tokens (loss_mask = 0)")
    print("GREEN = Assistant tokens (loss_mask = 1)")
    print("-" * 50)
    
    if len(input_ids) == 0 or len(loss_mask) == 0:
        print("Empty input")
        return
        
    current_mask = loss_mask[0].item()
    current_ids = []
    
    for i in range(len(input_ids)):
        if current_mask == loss_mask[i].item():
            current_ids.append(input_ids[i].item())
        else:
            if hasattr(tokenizer, 'decode'):
                decoded_text = tokenizer.decode(current_ids, skip_special_tokens=False)
            else:
                decoded_text = " ".join([f"token_{id}" for id in current_ids])
            if current_mask == 0:
                print(f"{RED}{decoded_text}{RESET}", end="")
            else:
                print(f"{GREEN}{decoded_text}{RESET}", end="")
            current_ids = [input_ids[i].item()]
            current_mask = loss_mask[i].item()
    
    # Print remaining tokens
    if current_ids:
        if hasattr(tokenizer, 'decode'):
            decoded_text = tokenizer.decode(current_ids, skip_special_tokens=False)
        else:
            decoded_text = " ".join([f"token_{id}" for id in current_ids])
        if current_mask == 0:
            print(f"{RED}{decoded_text}{RESET}")
        else:
            print(f"{GREEN}{decoded_text}{RESET}")
    print("\n" + "-" * 50)


if __name__ == "__main__":
    # Run a simple test to demonstrate functionality
    print("Running basic test of preprocess_conversations...")
    
    tokenizer = MockTokenizer()
    chat_template = TEMPLATE_REGISTRY.get("llama3")
    
    conversations = [
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What about Germany?"},
            {"role": "assistant", "content": "The capital of Germany is Berlin."},
        ]
    ]
    
    result = preprocess_conversations(
        tokenizer=tokenizer,
        conversations=conversations,
        chat_template=chat_template,
        max_length=512,
    )
    
    input_ids = result["input_ids"][0].squeeze(0)
    loss_mask = result["loss_mask"][0].squeeze(0)
    
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Loss mask shape: {loss_mask.shape}")
    print(f"Assistant tokens: {loss_mask.sum().item()}")
    print(f"Non-assistant tokens: {(loss_mask == 0).sum().item()}")
    
    # Visualize the loss mask
    visualize_loss_mask(tokenizer, input_ids, loss_mask)
    
    print("\nTest completed successfully!")
