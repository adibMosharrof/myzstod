import random

def generate_domain_examples(train_data):
    """
    Generates formatted API call examples for each unique domain and a default set from the given train dataset.

    Args:
        train_data: List of TodTurnApiCallCsvRow objects containing the train data.

    Returns:
        dict: A dictionary where keys are domains and values are formatted strings of examples.
    """
    # Initialize the result dictionary
    domain_examples = {}

    # Step 1: Extract all unique domain names separately
    unique_domains = set()
    for row in train_data:
        domains = row.domains.split(',')  # Split multi-domain entries
        unique_domains.update(domains)

    # Step 2: Process each domain
    for domain in unique_domains:
        # Filter data for the current domain (including multi-domain entries that contain it)
        domain_data = [row for row in train_data if domain in row.domains.split(',')]
        
        # Randomly select up to 5 unique dialog_ids
        dialog_ids = random.sample(set(row.dialog_id for row in domain_data), min(5, len(set(row.dialog_id for row in domain_data))))
        
        # Initialize a list to store formatted examples
        formatted_examples = []
        
        # Step 3: Process each dialog_id
        for dialog_id in dialog_ids:
            # Get rows for this dialog_id
            dialog_rows = [row for row in domain_data if row.dialog_id == dialog_id]
            
            # Filter rows where turn_row_type = 1
            filtered_rows = [row for row in dialog_rows if row.turn_row_type == 1]
            
            # Format examples
            for i, row in enumerate(filtered_rows, start=1):
                formatted_example = (
                    f"Example {i}: for this schema: {row.schema}\n"
                    f"and this context: {row.context}, here you have to make the API call like this: {row.target}."
                )
                formatted_examples.append(formatted_example)
        
        # Store the formatted examples for the domain
        domain_examples[domain] = "\n\n".join(formatted_examples)
    
    # Step 4: Process 'default' key (random dialog_ids across all domains)
    all_dialog_ids = set(row.dialog_id for row in train_data)
    random_dialog_ids = random.sample(all_dialog_ids, 5)
    
    # Initialize a list for 'default' examples
    default_examples = []
    
    for dialog_id in random_dialog_ids:
        dialog_rows = [row for row in train_data if row.dialog_id == dialog_id]
        filtered_rows = [row for row in dialog_rows if row.turn_row_type == 1]
        selected_rows = random.sample(filtered_rows, min(5, len(filtered_rows)))
        
        for i, row in enumerate(selected_rows, start=1):
            formatted_example = (
                f"Example {i}: for this schema: {row.schema}\n"
                f"and this context: {row.context}, here you have to make the API call like this: {row.target}."
            )
            default_examples.append(formatted_example)
    
    # Add 'default' examples to the dictionary
    domain_examples['default'] = "\n\n".join(default_examples)
    
    # Return the final dictionary
    return domain_examples
