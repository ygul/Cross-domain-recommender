import chat_orchestrator
orchestrator = chat_orchestrator.ChatOrchestrator()

while True:
    print("\nPlease let me know if you're looking for a book, TV series, and/or movie (type B, S, and/or M, multiple options are possible - ENTER for ALL):")
    q = input("> ")
    where_types = None
    if q:
        opts = set()
        for ch in q.upper():
            if ch == 'B':
                opts.add('Book')
            elif ch == 'S':
                opts.add('TV series')
            elif ch == 'M':
                opts.add('movie')
        where_types = opts if opts else None
    
    print("\nPlease describe your favorite book, series or movie (or 'exit' / ENTER to quit):")
    q = input("> ")
    if q in {"exit","quit", ""}: break
    
    print("\nSearching for relevant items...\n")
    print("-" * 40)
    print(orchestrator.run_once(q, item_types=where_types))
    print("-" * 40)
    