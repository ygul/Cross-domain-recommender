import chat_orchestrator
orchestrator = chat_orchestrator.ChatOrchestrator()

while True:
    print("Please describe your favorite book, series or movie (or 'exit' / ENTER to quit):")
    q = input("> ")
    if q in {"exit","quit", ""}: break
    print(orchestrator.run_once(q))
    