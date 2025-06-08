import ollama
import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor
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
    while len(selected) < min(3, len(models)):
        try:
            choice = int(input(f"Select model {len(selected) + 1} by number (0 to exit): "))
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


def model_inference(model_name, prompt):
    """Run model inference with performance tracking"""
    start_time = time.perf_counter()
    response_content = ""

    try:
        # Generate response with streaming
        stream = ollama.generate(
            model=model_name,
            prompt=prompt,
            stream=True
        )

        # Collect response without immediate printing
        for chunk in stream:
            token = chunk.get('response', '')
            response_content += token

        end_time = time.perf_counter()
        total_time = end_time - start_time
        token_count = len(response_content.split())
        avg_time_per_token = total_time / token_count if token_count > 0 else 0

        return {
            'model': model_name,
            'response': response_content,
            'total_time': total_time,
            'token_count': token_count,
            'avg_time_per_token': avg_time_per_token
        }

    except Exception as e:
        print(f"Error with model {model_name}: {e}")
        return {
            'model': model_name,
            'response': f"Error: {str(e)}",
            'total_time': 0,
            'token_count': 0,
            'avg_time_per_token': 0
        }


def display_responses(results):
    """Display model responses with performance metrics"""
    print("\n" + "=" * 80)
    print("MODEL RESPONSES")
    print("=" * 80)

    # Display responses first
    for res in results:
        print(f"\n[{res['model'].upper()}]")
        print(res['response'])
        print("-" * 60)

    # Create performance comparison table
    table_data = []
    for res in results:
        table_data.append([
            res['model'],
            f"{res['total_time']:.2f}s",
            res['token_count'],
            f"{res['avg_time_per_token'] * 1000:.2f}ms"
        ])

    print("\nPERFORMANCE COMPARISON:")
    print(tabulate(table_data,
                   headers=["Model", "Total Time", "Tokens", "Avg Time/Token"],
                   tablefmt="grid"))
    print("=" * 80 + "\n")


def chat_session(models):
    """Run interactive chat session with selected models"""
    print(f"\nStarting chat with {', '.join(models)}")
    print("Type 'exit' to quit or 'switch' to change models\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == 'exit':
            return False
        if user_input.lower() == 'switch':
            return True

        # Run models in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(model_inference, model, user_input) for model in models]
            results = [future.result() for future in futures]

        display_responses(results)


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