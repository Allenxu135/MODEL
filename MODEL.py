import ollama
import requests
import time
import threading
from tabulate import tabulate


def get_local_models():
    """Fetch locally available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = [model["name"] for model in response.json().get("models", [])]
        return models
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def select_models(models):
    """Prompt user to select models from available list"""
    if not models:
        print("No models available. Please download models first.")
        return None

    print("\nAvailable models:")
    for i, model in enumerate(models):
        print(f"{i + 1}. {model}")

    selected = []
    while len(selected) < 3:
        try:
            choice = int(input(f"\nSelect model {len(selected) + 1} by number (0 to exit): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(models):
                model_name = models[choice - 1]
                if model_name not in selected:
                    selected.append(model_name)
                    print(f"Added {model_name}")
                else:
                    print("Model already selected. Choose another.")
            else:
                print("Invalid selection. Try again.")
        except ValueError:
            print("Please enter a valid number.")

    return selected


def get_model_response(model_name, prompt, results, index):
    """Get model response with performance metrics"""
    start_time = time.time()
    token_times = []
    response_content = ""

    try:
        stream = ollama.generate(
            model=model_name,
            prompt=prompt,
            stream=True
        )

        for chunk in stream:
            token = chunk.get('response', '')
            token_time = time.time()
            response_content += token
            if token:  # Only record time for actual tokens
                token_times.append((token, token_time - start_time))

        end_time = time.time()
        total_time = end_time - start_time
        token_count = len(response_content.split())
        avg_time_per_token = total_time / token_count if token_count > 0 else 0

        results[model_name] = {
            'response': response_content,
            'total_time': total_time,
            'token_count': token_count,
            'avg_time_per_token': avg_time_per_token,
            'token_times': token_times
        }

    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        results[model_name] = {
            'response': f"Error: {str(e)}",
            'total_time': 0,
            'token_count': 0,
            'avg_time_per_token': 0,
            'token_times': []
        }


def display_results(results):
    """Display model responses with performance metrics"""
    print("\n" + "=" * 80)
    print("MODEL RESPONSES WITH PERFORMANCE METRICS")
    print("=" * 80)

    for model, data in results.items():
        print(f"\n[ {model.upper()} ]")
        print(
            f"Response time: {data['total_time']:.2f}s | Tokens: {data['token_count']} | Avg: {data['avg_time_per_token'] * 1000:.2f}ms/token")
        print("-" * 60)
        print(data['response'])
        print("-" * 60)

    # Create performance comparison table
    table_data = []
    for model, data in results.items():
        table_data.append([
            model,
            f"{data['total_time']:.2f}s",
            data['token_count'],
            f"{data['avg_time_per_token'] * 1000:.2f}ms"
        ])

    print("\nPERFORMANCE COMPARISON:")
    print(tabulate(table_data,
                   headers=["Model", "Total Time", "Tokens", "Avg Time/Token"],
                   tablefmt="grid"))

    print("=" * 80 + "\n")


def chat_session(models):
    """Run interactive chat session with selected models"""
    histories = {model: [] for model in models}

    print(f"\nStarting chat with {', '.join(models)}")
    print("Type 'exit' to quit or 'switch' to change models\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            return False
        if user_input.lower() == 'switch':
            return True

        results = {}
        threads = []

        # Start threads for each model
        for model in models:
            # Add user input to model's history
            histories[model].append({"role": "user", "content": user_input})

            t = threading.Thread(
                target=get_model_response,
                args=(model, user_input, results, models.index(model))
            )
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Add responses to histories and display
        for model in models:
            if model in results:
                histories[model].append({
                    "role": "assistant",
                    "content": results[model]['response']
                })

        display_results(results)


def main():
    """Main application loop"""
    print("Ollama Multi-Model Chat Interface")
    print("=" * 50)

    models = get_local_models()
    if not models:
        print("No models found. Please download models first with 'ollama pull <model>'.")
        return

    selected_models = select_models(models)
    if not selected_models:
        return

    while True:
        switch_requested = chat_session(selected_models)
        if not switch_requested:
            break

        # If user requested switch, reload models and reselect
        models = get_local_models()
        selected_models = select_models(models)
        if not selected_models:
            break


if __name__ == "__main__":
    main()