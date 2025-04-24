from voice_assistant import VoiceAssistant

def main():
    # Initialize the voice assistant
    assistant = VoiceAssistant()
    
    while True:
        print("\nVoice Assistant Menu:")
        print("1. Register new user")
        print("2. Process voice command")
        print("3. Exit")
        
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            user_id = input("Enter a unique user ID: ")
            if assistant.register_new_user(user_id):
                print(f"Successfully registered user: {user_id}")
            else:
                print("Failed to register user. Please try again.")
                
        elif choice == '2':
            print("Processing voice command...")
            assistant.process_voice_command()
            
        elif choice == '3':
            print("Exiting voice assistant...")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()