import pytest
import torch
import unittest
import os
from unittest.mock import Mock, patch

from specforge.data.preprocessing import preprocess_conversations
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate
from transformers import AutoTokenizer

class TestPreprocessConversations(unittest.TestCase):
    """Test suite for preprocess_conversations function."""

    def test_preprocess_conversations_basic(self):
        """Test basic conversation preprocessing."""
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
            
            # Create example chat template
            chat_template = ChatTemplate(
                assistant_header="<|start_header_id|>assistant<|end_header_id|>\n\n",
                user_header="<|start_header_id|>user<|end_header_id|>",
                system_prompt="You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                end_of_turn_token="<|eot_id|>",
            ),

            # Run the function
            result = preprocess_conversations(
                tokenizer=mock_tokenizer,
                conversations=conversations,
                chat_template=chat_template,
                max_length=512,
            )
            
            # Assertions
            self.assertIn("input_ids", result)
            self.assertIn("loss_mask", result)
            self.assertIn("attention_mask", result)
            self.assertEqual(len(result["input_ids"]), 1)
            self.assertEqual(len(result["loss_mask"]), 1)
            self.assertEqual(len(result["attention_mask"]), 1)
            
            # Check tensor shapes
            input_ids = result["input_ids"][0]
            loss_mask = result["loss_mask"][0]
            attention_mask = result["attention_mask"][0]
            
            self.assertEqual(input_ids.shape, loss_mask.shape)
            self.assertEqual(input_ids.shape, attention_mask.shape)
            self.assertEqual(len(input_ids.shape), 2)  # Should have batch dimension


    def test_preprocess_conversations_multiple(self):
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
            self.assertEqual(len(result["input_ids"]), 2)
            self.assertEqual(len(result["loss_mask"]), 2)
            self.assertEqual(len(result["attention_mask"]), 2)


    def test_preprocess_conversations_empty(self):
        """Test preprocessing with empty conversations list."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            
            def mock_apply_chat_template(messages, tokenize=False, add_generation_prompt=False):
                return ""
            
            mock_tokenizer.apply_chat_template = mock_apply_chat_template
            mock_tokenizer_class.return_value = mock_tokenizer
            
            conversations = []
            
            chat_template = TEMPLATE_REGISTRY.get("llama3")
            
            result = preprocess_conversations(
                tokenizer=mock_tokenizer,
                conversations=conversations,
                chat_template=chat_template,
                max_length=512,
            )
            
            # Should return empty lists
            self.assertEqual(result["input_ids"], [])
            self.assertEqual(result["loss_mask"], [])
            self.assertEqual(result["attention_mask"], [])


    def test_preprocess_conversations_role_validation(self):
        """Test role validation."""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer_class:
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token_id = 0
            
            # We need to mock the tokenizer but the actual validation happens before tokenization
            mock_tokenizer_class.return_value = mock_tokenizer
            
            conversations = [
                [
                    {"role": "user", "content": "First message"},
                    {"role": "user", "content": "Second message"},  # Invalid: two user messages
                ]
            ]
            
            chat_template = TEMPLATE_REGISTRY.get("llama3")
            
            # This should raise an AssertionError
            with self.assertRaises(AssertionError):
                preprocess_conversations(
                    tokenizer=mock_tokenizer,
                    conversations=conversations,
                    chat_template=chat_template,
                    max_length=512,
                )


    def test_preprocess_conversations_none(self):
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
            self.assertEqual(len(result["input_ids"]), 1)


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
    
    # Handle both 1D and 2D tensors - flatten if needed
    if input_ids.dim() > 1:
        input_ids = input_ids.flatten()
    if loss_mask.dim() > 1:
        loss_mask = loss_mask.flatten()
    
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
    # Run unit tests
    unittest.main(verbosity=2)

    # # Visualize loss mask for conversations
    # # model_path = "Qwen/Qwen3-4B"
    # model_path = "/shared/public/elr-models/Qwen/Qwen3-4B/1cfa9a7208912126459214e8b04321603b3df60c"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # chat_template = TEMPLATE_REGISTRY.get("qwen")

    # # Using conversations list
    # conversations = [
    #     [
    #         {"role": "user", "content": "What is 2+2?"},
    #         {"role": "assistant", "content": "The answer is 4."},
    #         {"role": "user", "content": "I don't think that's right"},
    #         {"role": "assistant", "content": "I'm pretty sure it's 4."}
    #     ],
    #     [
    #         {"role": "user", "content": "How do you boil water?"},
    #         {"role": "assistant", "content": "Use a stove."}
    #     ]
    # ]
    # results = preprocess_conversations(
    #     tokenizer=tokenizer,
    #     conversations=conversations,
    #     chat_template=chat_template,
    #     max_length=512,
    #     is_preformatted=False
    # )
    # for i in range(len(results["input_ids"])):
    #     visualize_loss_mask(tokenizer, results['input_ids'][i], results['loss_mask'][i])

    # # Using preformatted conversation
    # preformatted_conversations = [
    #     "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\nThe answer is 4.<|im_end|>\n<|im_start|>user\nI don't think that's right<|im_end|>\n<|im_start|>assistant\n<think>\nI know 2+2 is 4</think>\n\nI'm pretty sure it's 4.<|im_end|>\n",
    # ]
    # results = preprocess_conversations(
    #     tokenizer=tokenizer,
    #     conversations=preformatted_conversations,
    #     chat_template=chat_template,
    #     max_length=512,
    #     is_preformatted=True
    # )
    # for i in range(len(results["input_ids"])):
    #     visualize_loss_mask(tokenizer, results['input_ids'][i], results['loss_mask'][i])




