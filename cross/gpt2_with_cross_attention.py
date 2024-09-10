from transformers import GPT2Model, GPT2Config, GPT2Tokenizer, AutoModel, AutoConfig
import torch

# Step 1: Modify the configuration
config = AutoConfig.from_pretrained("gpt2")
config.add_cross_attention = True  # Enable cross-attention

# Step 2: Instantiate the model with the modified configuration
model = AutoModel.from_config(config)

encoder_model = AutoModel.from_pretrained("gpt2")
# Example usage
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
input_text = "The quick brown fox jumps over the lazy dog."
input_text = """You are an expert chat assistant for the domain: restaurants.
Instructions: As an expert, you must generate the most appropriate response for the chat assistant.
The response can be an api call or a response to the user.
You will be provided with a Schema for domain: restaurants.
You will be provided an incomplete dialog between a user and a chat assistant, and an optional search results.
Dialog History:
User: Hi, could you get me a restaurant booking on the 8th please?
System: Any preference on the restaurant, location and time?
User: Could you get me a reservation at P.f. Chang's in Corte Madera at afternoon 12?
System: Please confirm your reservation at P.f. Chang's in Corte Madera at 12 pm for 2 on March 8th.
Last User Utterance:Sure, that is great.
End Dialog History
. Using the Dialog History, Search Results, and by following the Instructions please generate the response for the chat assistant."""

schema_txt = """
Schema for Restaurants
Intent: ReserveRestaurant
required slots: restaurant name,location,time
optional slots: number of seats,date
Intent: FindRestaurants
required slots: category,location
optional slots: price range,has vegetarian options,has seating outdoors
"""

input_ids = tokenizer(input_text, return_tensors="pt").input_ids
schema_ids = tokenizer(schema_txt, return_tensors="pt").input_ids
# Example encoder_hidden_states (could be from another model or a previous layer)
# encoder_hidden_states = torch.randn(1, input_ids.size(-1), config.hidden_size)
with torch.no_grad():
    outputs = encoder_model(input_ids=schema_ids)
# encoder_hidden_states = outputs.last_hidden_state[:, 0, :]
encoder_hidden_states = outputs.last_hidden_state


# Forward pass with cross-attention
outputs = model(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states)

print(
    outputs.last_hidden_state.shape
)  # Output shape should match the input sequence length and hidden size
