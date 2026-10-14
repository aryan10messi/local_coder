"""
Training data generator for fine-tuning Qwen2.5-Coder-3B-Instruct on tool calling.

Generates execution-verified JSONL examples covering:
  1. Tool usage (all 7 tools with diverse phrasings)
  2. No-tool responses (chat, greetings, concept questions)
  3. Post-tool summarization (multi-turn: tool result → summary → STOP)
  4. Error recovery (tool fails → try different approach)
  5. Multi-step tasks (read → edit, grep → read, create → run)
  6. Tool selection (picking the RIGHT tool for ambiguous requests)

Every create_file/write_file example has its code content verified by Python's
compile() before inclusion. Bad code is rejected, not trained on.

Output format per example:
  {"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "reasoning + [TOOL_CALL]JSON[/TOOL_CALL]"}
  ]}
"""

import json
import random
import os
import sys
import textwrap


SYSTEM_PROMPT = """You are a code assistant. You have these tools:
- read_file(path): Read a file
- edit_file(path, old_text, new_text): Replace text in a file
- create_file(path, content): Create a new file
- write_file(path, content): Overwrite a file
- grep_codebase(pattern, directory): Search files
- list_files(directory): List directory contents
- run_command(command): Run a shell command

To use a tool, output EXACTLY this format:
[TOOL_CALL]{"name": "TOOL_NAME", "parameters": {"param1": "value1"}}[/TOOL_CALL]

Example - creating a file:
I'll create the file.
[TOOL_CALL]{"name": "create_file", "parameters": {"path": "hello.py", "content": "print('hello')"}}[/TOOL_CALL]

Example - reading a file:
Let me read that file.
[TOOL_CALL]{"name": "read_file", "parameters": {"path": "main.py"}}[/TOOL_CALL]

Rules:
- Use ONLY the [TOOL_CALL] JSON format shown above. No other format.
- One tool call per response.
- After a tool succeeds, tell the user what happened and STOP.
- If a tool errors, try a different approach.
- If the user is just chatting (greetings, questions about concepts), respond normally without using tools."""


def tool_call(name: str, params: dict) -> str:
    return f'[TOOL_CALL]{json.dumps({"name": name, "parameters": params})}[/TOOL_CALL]'


def verify_python(code: str) -> bool:
    try:
        compile(code, "<test>", "exec")
        return True
    except SyntaxError:
        return False


def msg(role, content):
    return {"role": role, "content": content}


def example(*messages):
    return {"messages": list(messages)}


# ---------------------------------------------------------------------------
# NO-TOOL RESPONSES — greetings, concepts, chitchat
# ---------------------------------------------------------------------------
def no_tool_examples():
    pairs = [
        ("hi", "Hey! How can I help you with your code today?"),
        ("hello", "Hello! What are you working on?"),
        ("hey there", "Hey! What can I help you with?"),
        ("what's up", "Not much! Ready to help with any code tasks you have."),
        ("good morning", "Good morning! What are we working on today?"),
        ("thanks", "You're welcome! Let me know if you need anything else."),
        ("thank you that was helpful", "Glad I could help! Feel free to ask if anything else comes up."),
        ("bye", "See you later! Good luck with your project."),
        ("what is a decorator in python?",
         "A decorator is a function that wraps another function to extend its behavior without modifying it. You define one with `@decorator_name` above a function. Common uses include logging, authentication checks, and caching."),
        ("explain list comprehensions",
         "List comprehensions are a concise way to create lists. Instead of writing a for loop, you write `[expression for item in iterable if condition]`. For example, `[x**2 for x in range(10) if x % 2 == 0]` gives you squares of even numbers."),
        ("what's the difference between a list and a tuple?",
         "Lists are mutable (you can change them after creation) and use square brackets `[]`. Tuples are immutable (fixed after creation) and use parentheses `()`. Use tuples for data that shouldn't change, like coordinates or database records."),
        ("how do virtual environments work?",
         "A virtual environment is an isolated Python installation. When you activate one, `pip install` puts packages in that environment's directory instead of globally. This lets different projects use different package versions without conflicts. Create one with `python3 -m venv .venv` and activate with `source .venv/bin/activate`."),
        ("what does __init__.py do?",
         "It marks a directory as a Python package so you can import from it. It can be empty, or it can define what gets exported when someone does `from package import *`. In modern Python (3.3+), implicit namespace packages exist without it, but it's still the convention."),
        ("what's a good project structure for a flask app?",
         "A common layout is: `app/` for your Flask code with `__init__.py`, `models.py`, `routes.py`, and `templates/`. Then `tests/` for tests, `config.py` at the root, and `requirements.txt`. For larger apps, use blueprints and split routes into separate files."),
        ("can you explain what an API is?",
         "An API (Application Programming Interface) is a way for programs to talk to each other. A web API exposes endpoints (URLs) that accept requests and return data, usually as JSON. For example, a weather API might have an endpoint like `/api/weather?city=SF` that returns temperature data."),
        ("what does async/await do in python?",
         "async/await lets you write concurrent code that doesn't block while waiting for I/O operations like network requests or file reads. You mark a function with `async def` and use `await` to pause it until an operation completes. The event loop runs other tasks during that pause."),
        ("how does git branching work?",
         "A branch is a separate line of development. When you create a branch with `git checkout -b feature`, you get a copy of the current state to work on independently. When you're done, you merge it back into main. This lets multiple people work on different features without stepping on each other."),
        ("should I use classes or functions for this project?",
         "It depends on whether you have shared state. If your functions all operate on the same data and need to maintain state between calls, a class groups that nicely. If each operation is independent and stateless, plain functions are simpler and easier to test. Start with functions and refactor to classes if you find yourself passing the same data everywhere."),
        ("what is a REST API?",
         "REST is a set of conventions for web APIs. Resources are identified by URLs (like `/users/42`), and you use HTTP methods to act on them: GET to read, POST to create, PUT to update, DELETE to remove. Responses are typically JSON. It's the most common API style for web applications."),
        ("I'm stuck on a bug and don't know where to start",
         "Start by reproducing the bug consistently. Then narrow it down: check the error message and traceback, add print statements or use a debugger to see what values are wrong, and work backwards from the crash to find where the data goes bad. If you share the error or the code, I can help pinpoint it."),
        ("what does pip freeze do?",
         "It prints every installed package and its exact version, one per line. The output format matches requirements.txt, so `pip freeze > requirements.txt` captures your current environment. This lets others recreate the same setup with `pip install -r requirements.txt`."),
        ("is python good for web development?",
         "Yes, Python is great for web backends. Django is a batteries-included framework for larger apps, and Flask/FastAPI are lighter for APIs and microservices. FastAPI is especially popular for its speed and automatic OpenAPI docs. Python isn't used for frontend though — that's JavaScript/TypeScript territory."),
        ("what are type hints in python?",
         "Type hints let you annotate what types your functions expect and return, like `def greet(name: str) -> str:`. Python doesn't enforce them at runtime, but tools like mypy can check them statically. They also improve IDE autocomplete and make code self-documenting."),
        ("how do I handle errors in python?",
         "Use try/except blocks. Put the code that might fail in `try:`, and handle specific exceptions in `except ValueError as e:`. Always catch specific exceptions rather than bare `except:`. Use `finally:` for cleanup that should run regardless, like closing files."),
        ("what's the point of requirements.txt?",
         "It lists your project's dependencies so anyone can install them with one command: `pip install -r requirements.txt`. Pin versions (like `flask==3.0.0`) for reproducibility. Without it, collaborators have to guess which packages your project needs."),
        ("explain what a class method vs static method is",
         "A `@classmethod` receives the class itself as its first argument (`cls`), so it can access class-level attributes and create instances. A `@staticmethod` receives no implicit arguments — it's just a regular function that lives inside the class for organizational purposes. Use classmethod for alternate constructors, staticmethod for utility functions."),
        ("yo", "Hey! What are you working on?"),
        ("sup", "Hey! Ready to help — what do you need?"),
        ("whats good", "All good here! What can I help you build?"),
        ("hola", "Hola! What are we coding today?"),
        ("how are you", "Doing well! What can I help you with?"),
        ("help", "I can help you read, write, edit, and search code files, plus run shell commands. Just tell me what you need!"),
        ("what can you do", "I can read and write files, search your codebase, edit specific parts of files, and run shell commands. Tell me what you're working on and I'll help out."),
        ("never mind", "No problem! Let me know when you need something."),
        ("ok cool", "Sounds good! I'm here if you need anything else."),
    ]
    return [
        example(msg("system", SYSTEM_PROMPT), msg("user", u), msg("assistant", a))
        for u, a in pairs
    ]


# ---------------------------------------------------------------------------
# READ_FILE examples
# ---------------------------------------------------------------------------
def read_file_examples():
    pairs = [
        ("Show me what's in main.py", "I'll read that file for you."),
        ("Can you open src/utils/helpers.py? I need to see the code", "Let me read that file."),
        ("What does the config file look like?", "I'll read the config file to show you its contents."),
        ("cat server.js", "I'll read server.js for you."),
        ("I want to see the contents of tests/test_auth.py", "Let me pull up that test file."),
        ("Read the README", "I'll read the README file."),
        ("What's inside package.json?", "Let me check the package.json."),
        ("Open the database migration file at db/migrations/001_create_users.sql", "I'll read that migration file."),
        ("Show me the Dockerfile", "Let me read the Dockerfile."),
        ("I need to review app/models/user.py before making changes", "I'll read the file so we can review it together."),
        ("Pull up the .env.example file", "Let me read that file."),
        ("What's in the requirements.txt?", "I'll check the requirements file."),
        ("Can you read lib/api_client.rb for me?", "Let me read that Ruby file."),
        ("I forgot what I put in setup.py, can you show me?", "I'll read setup.py for you."),
        ("Let me see the Makefile", "I'll open the Makefile."),
        ("Show me the nginx config at deploy/nginx.conf", "Let me read that configuration file."),
        ("What does src/components/Header.tsx look like?", "I'll read that React component."),
        ("Read the CHANGELOG", "Let me pull up the changelog."),
        ("I want to check the CI config at .github/workflows/test.yml", "I'll read that GitHub Actions workflow."),
        ("Show me the error handler in middleware/errors.py", "I'll read that middleware file."),
        ("yo show me main.py real quick", "I'll read main.py for you."),
        ("main.py", "I'll read main.py for you."),
        ("I wanna see main.py", "I'll read main.py for you."),
        ("What's main.py look like?", "I'll read main.py for you."),
        ("Print main.py", "I'll read main.py for you."),
        ("Display the code in main.py", "I'll read main.py for you."),
    ]
    paths = [
        "main.py", "src/utils/helpers.py", "config.py", "server.js",
        "tests/test_auth.py", "README.md", "package.json",
        "db/migrations/001_create_users.sql", "Dockerfile", "app/models/user.py",
        ".env.example", "requirements.txt", "lib/api_client.rb", "setup.py",
        "Makefile", "deploy/nginx.conf", "src/components/Header.tsx",
        "CHANGELOG.md", ".github/workflows/test.yml", "middleware/errors.py",
        "main.py", "main.py", "main.py", "main.py", "main.py", "main.py",
    ]
    return [
        example(
            msg("system", SYSTEM_PROMPT),
            msg("user", u),
            msg("assistant", f"{r}\n\n{tool_call('read_file', {'path': p})}"),
        )
        for (u, r), p in zip(pairs, paths)
    ]


# ---------------------------------------------------------------------------
# CREATE_FILE examples (all code verified by compile())
# ---------------------------------------------------------------------------
def create_file_examples():
    items = [
        (
            "Create a hello world Python script",
            "I'll create a simple hello world script.",
            "hello.py",
            'print("Hello, world!")\n',
        ),
        (
            "Make a new file called utils.py with a function that checks if a number is prime",
            "I'll create a utility file with a prime-checking function.",
            "utils.py",
            textwrap.dedent("""\
                def is_prime(n):
                    if n < 2:
                        return False
                    for i in range(2, int(n**0.5) + 1):
                        if n % i == 0:
                            return False
                    return True
            """),
        ),
        (
            "Create a basic Flask app in app.py",
            "I'll set up a basic Flask application.",
            "app.py",
            textwrap.dedent("""\
                from flask import Flask, jsonify

                app = Flask(__name__)


                @app.route('/')
                def index():
                    return jsonify({'message': 'Hello from Flask'})


                @app.route('/health')
                def health():
                    return jsonify({'status': 'ok'})


                if __name__ == '__main__':
                    app.run(debug=True)
            """),
        ),
        (
            "Write a test file for the user model at tests/test_user.py",
            "I'll create a test file for the user model.",
            "tests/test_user.py",
            textwrap.dedent("""\
                import pytest
                from models.user import User


                class TestUser:
                    def test_create_user(self):
                        user = User(name='Alice', email='alice@example.com')
                        assert user.name == 'Alice'
                        assert user.email == 'alice@example.com'

                    def test_user_str(self):
                        user = User(name='Bob', email='bob@example.com')
                        assert 'Bob' in str(user)
            """),
        ),
        (
            "Make a Python script at scripts/seed_db.py that inserts sample data",
            "I'll create a database seeding script.",
            "scripts/seed_db.py",
            textwrap.dedent("""\
                import sqlite3


                def seed():
                    conn = sqlite3.connect('app.db')
                    cur = conn.cursor()
                    cur.execute(
                        'INSERT INTO users (name, email) VALUES (?, ?)',
                        ('Alice', 'alice@example.com'),
                    )
                    cur.execute(
                        'INSERT INTO users (name, email) VALUES (?, ?)',
                        ('Bob', 'bob@example.com'),
                    )
                    conn.commit()
                    conn.close()
                    print('Seeded database')


                if __name__ == '__main__':
                    seed()
            """),
        ),
        (
            "I need a conftest.py in the tests folder with a database fixture",
            "I'll create the conftest with a database fixture.",
            "tests/conftest.py",
            textwrap.dedent("""\
                import pytest
                import sqlite3


                @pytest.fixture
                def db():
                    conn = sqlite3.connect(':memory:')
                    cur = conn.cursor()
                    cur.execute(
                        'CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)'
                    )
                    conn.commit()
                    yield conn
                    conn.close()
            """),
        ),
        (
            "Create an __init__.py in the models directory",
            "I'll create the package init file.",
            "models/__init__.py",
            textwrap.dedent("""\
                from .user import User
                from .post import Post

                __all__ = ['User', 'Post']
            """),
        ),
        (
            "Write a logging config at config/logging.py",
            "I'll create a logging configuration module.",
            "config/logging.py",
            textwrap.dedent("""\
                import logging
                import sys


                def setup_logging(level='INFO'):
                    handler = logging.StreamHandler(sys.stdout)
                    handler.setFormatter(
                        logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
                    )
                    root = logging.getLogger()
                    root.setLevel(getattr(logging, level))
                    root.addHandler(handler)
            """),
        ),
        (
            "Create a constants file at config/constants.py",
            "I'll create a constants module.",
            "config/constants.py",
            textwrap.dedent("""\
                MAX_RETRIES = 3
                TIMEOUT_SECONDS = 30
                PAGE_SIZE = 25
                MAX_UPLOAD_SIZE_MB = 10
                ALLOWED_EXTENSIONS = {'.jpg', '.png', '.gif', '.pdf'}
                DATE_FORMAT = '%Y-%m-%d'
                DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
            """),
        ),
        (
            "Create a middleware file at middleware/auth.py that checks for an API key header",
            "I'll create an authentication middleware.",
            "middleware/auth.py",
            textwrap.dedent("""\
                import os
                from functools import wraps
                from flask import request, jsonify


                def require_api_key(f):
                    @wraps(f)
                    def decorated(*args, **kwargs):
                        api_key = request.headers.get('X-API-Key')
                        expected = os.environ.get('API_KEY')
                        if not api_key or api_key != expected:
                            return jsonify({'error': 'Unauthorized'}), 401
                        return f(*args, **kwargs)
                    return decorated
            """),
        ),
        (
            "Write a data class for a blog post in models/post.py",
            "I'll create a Post data class.",
            "models/post.py",
            textwrap.dedent("""\
                from dataclasses import dataclass, field
                from datetime import datetime


                @dataclass
                class Post:
                    title: str
                    body: str
                    author: str
                    created_at: datetime = field(default_factory=datetime.now)
                    published: bool = False

                    def publish(self):
                        self.published = True

                    def summary(self, length=100):
                        if len(self.body) <= length:
                            return self.body
                        return self.body[:length].rsplit(' ', 1)[0] + '...'
            """),
        ),
        (
            "plz make a hello world script",
            "I'll create a hello world script.",
            "hello.py",
            'print("Hello, world!")\n',
        ),
        (
            "Write me a simple Python hello world and save it",
            "I'll create a hello world script.",
            "hello.py",
            'print("Hello, world!")\n',
        ),
        (
            "I need a hello.py that prints hello world",
            "I'll create that for you.",
            "hello.py",
            'print("Hello, world!")\n',
        ),
        (
            "Generate a basic hello world Python file",
            "I'll create a hello world script.",
            "hello.py",
            'print("Hello, world!")\n',
        ),
        (
            "Create a number guessing game in guess.py",
            "I'll create a number guessing game.",
            "guess.py",
            textwrap.dedent("""\
                import random


                def play():
                    number = random.randint(1, 10)
                    print('Guess a number between 1 and 10!')

                    while True:
                        try:
                            guess = int(input('Your guess: '))
                        except ValueError:
                            print('Please enter a number.')
                            continue

                        if guess < number:
                            print('Too low!')
                        elif guess > number:
                            print('Too high!')
                        else:
                            print('Correct! You got it!')
                            break


                if __name__ == '__main__':
                    play()
            """),
        ),
        (
            "Make a todo list app in todo.py",
            "I'll create a simple command-line todo app.",
            "todo.py",
            textwrap.dedent("""\
                todos = []


                def show():
                    if not todos:
                        print('No todos yet.')
                        return
                    for i, item in enumerate(todos, 1):
                        status = 'x' if item['done'] else ' '
                        print(f'  [{status}] {i}. {item["text"]}')


                def add(text):
                    todos.append({'text': text, 'done': False})
                    print(f'Added: {text}')


                def complete(index):
                    if 1 <= index <= len(todos):
                        todos[index - 1]['done'] = True
                        print(f'Completed: {todos[index - 1]["text"]}')
                    else:
                        print('Invalid index.')


                def main():
                    print('Todo List (type help for commands)')
                    while True:
                        cmd = input('> ').strip()
                        if cmd == 'quit':
                            break
                        elif cmd == 'help':
                            print('Commands: show, add <text>, done <n>, quit')
                        elif cmd == 'show':
                            show()
                        elif cmd.startswith('add '):
                            add(cmd[4:])
                        elif cmd.startswith('done '):
                            try:
                                complete(int(cmd[5:]))
                            except ValueError:
                                print('Usage: done <number>')
                        else:
                            print('Unknown command. Type help.')


                if __name__ == '__main__':
                    main()
            """),
        ),
        (
            "Create a simple calculator script",
            "I'll create a calculator script.",
            "calculator.py",
            textwrap.dedent("""\
                def calculate(a, op, b):
                    if op == '+':
                        return a + b
                    elif op == '-':
                        return a - b
                    elif op == '*':
                        return a * b
                    elif op == '/':
                        if b == 0:
                            return 'Error: division by zero'
                        return a / b
                    else:
                        return f'Unknown operator: {op}'


                def main():
                    print('Simple Calculator')
                    print('Enter expressions like: 2 + 3')
                    print('Type quit to exit')

                    while True:
                        expr = input('> ').strip()
                        if expr == 'quit':
                            break
                        parts = expr.split()
                        if len(parts) != 3:
                            print('Format: <number> <operator> <number>')
                            continue
                        try:
                            a = float(parts[0])
                            b = float(parts[2])
                        except ValueError:
                            print('Invalid numbers.')
                            continue
                        result = calculate(a, parts[1], b)
                        print(f'= {result}')


                if __name__ == '__main__':
                    main()
            """),
        ),
        (
            "Make a file counter script that counts lines in Python files",
            "I'll create a line counter script.",
            "line_count.py",
            textwrap.dedent("""\
                import os
                import sys


                def count_lines(directory):
                    total = 0
                    files = 0
                    for root, dirs, filenames in os.walk(directory):
                        dirs[:] = [d for d in dirs if d not in {'__pycache__', '.venv', 'venv', '.git'}]
                        for name in filenames:
                            if name.endswith('.py'):
                                path = os.path.join(root, name)
                                with open(path) as f:
                                    lines = sum(1 for line in f)
                                print(f'  {path}: {lines}')
                                total += lines
                                files += 1
                    return files, total


                if __name__ == '__main__':
                    target = sys.argv[1] if len(sys.argv) > 1 else '.'
                    files, total = count_lines(target)
                    print(f'\\n{files} files, {total} lines total')
            """),
        ),
        (
            "Create a simple HTTP server script at server.py",
            "I'll create a simple HTTP server.",
            "server.py",
            textwrap.dedent("""\
                from http.server import HTTPServer, SimpleHTTPRequestHandler
                import sys


                def run(port=8000):
                    server = HTTPServer(('', port), SimpleHTTPRequestHandler)
                    print(f'Serving on http://localhost:{port}')
                    try:
                        server.serve_forever()
                    except KeyboardInterrupt:
                        print('\\nShutting down.')
                        server.shutdown()


                if __name__ == '__main__':
                    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
                    run(port)
            """),
        ),
        (
            "Create a JSON file reader at json_reader.py",
            "I'll create a JSON file reader utility.",
            "json_reader.py",
            textwrap.dedent("""\
                import json
                import sys


                def read_json(path):
                    with open(path) as f:
                        data = json.load(f)
                    return data


                def pretty_print(data, indent=2):
                    print(json.dumps(data, indent=indent, default=str))


                if __name__ == '__main__':
                    if len(sys.argv) < 2:
                        print('Usage: python json_reader.py <file.json>')
                        sys.exit(1)
                    data = read_json(sys.argv[1])
                    pretty_print(data)
            """),
        ),
    ]

    results = []
    verified = 0
    rejected = 0
    for user_msg, reasoning, path, content in items:
        if path.endswith(".py"):
            if not verify_python(content):
                print(f"  REJECTED (syntax error): {path} for '{user_msg[:50]}'")
                rejected += 1
                continue
            verified += 1
        results.append(example(
            msg("system", SYSTEM_PROMPT),
            msg("user", user_msg),
            msg("assistant", f"{reasoning}\n\n{tool_call('create_file', {'path': path, 'content': content})}"),
        ))
    print(f"  create_file: {verified} verified, {rejected} rejected")
    return results


# ---------------------------------------------------------------------------
# WRITE_FILE examples (overwrite / full rewrite — all code verified)
# ---------------------------------------------------------------------------
def write_file_examples():
    items = [
        (
            "Rewrite main.py to use argparse instead of sys.argv",
            "I'll rewrite main.py with argparse-based argument parsing.",
            "main.py",
            textwrap.dedent("""\
                import argparse


                def main():
                    parser = argparse.ArgumentParser(description='My CLI tool')
                    parser.add_argument('input', help='Input file path')
                    parser.add_argument('-o', '--output', default='out.txt', help='Output file')
                    parser.add_argument('-v', '--verbose', action='store_true')
                    args = parser.parse_args()

                    if args.verbose:
                        print(f'Processing {args.input}')

                    with open(args.input) as f:
                        data = f.read()

                    with open(args.output, 'w') as f:
                        f.write(data)
                    print(f'Written to {args.output}')


                if __name__ == '__main__':
                    main()
            """),
        ),
        (
            "Replace the entire config.py with a cleaner version using dataclasses",
            "I'll rewrite config.py using a dataclass.",
            "config.py",
            textwrap.dedent("""\
                import os
                from dataclasses import dataclass


                @dataclass
                class Config:
                    debug: bool = False
                    database_url: str = 'sqlite:///app.db'
                    secret_key: str = 'change-me'
                    port: int = 8080

                    @classmethod
                    def from_env(cls):
                        return cls(
                            debug=os.environ.get('DEBUG', 'false').lower() == 'true',
                            database_url=os.environ.get('DATABASE_URL', cls.database_url),
                            secret_key=os.environ.get('SECRET_KEY', cls.secret_key),
                            port=int(os.environ.get('PORT', cls.port)),
                        )


                config = Config.from_env()
            """),
        ),
        (
            "The requirements.txt is a mess. Rewrite it cleanly with pinned versions",
            "I'll rewrite requirements.txt with clean, pinned versions.",
            "requirements.txt",
            "flask==3.1.0\nrequests==2.32.3\nsqlalchemy==2.0.36\npydantic==2.10.3\npytest==8.3.4\nruff==0.8.4\n",
        ),
        (
            "Completely redo the README with proper documentation",
            "I'll rewrite the README with clear documentation.",
            "README.md",
            textwrap.dedent("""\
                # MyProject

                A brief description of what this project does.

                ## Setup

                ```bash
                python3 -m venv .venv
                source .venv/bin/activate
                pip install -r requirements.txt
                ```

                ## Usage

                ```bash
                python3 main.py input.txt -o output.txt
                ```

                ## Testing

                ```bash
                pytest tests/ -v
                ```

                ## License

                MIT
            """),
        ),
        (
            "Overwrite the Dockerfile — the current one is wrong",
            "I'll rewrite the Dockerfile with a correct multi-stage build.",
            "Dockerfile",
            textwrap.dedent("""\
                FROM python:3.12-slim AS builder
                WORKDIR /app
                COPY requirements.txt .
                RUN pip install --no-cache-dir -r requirements.txt

                FROM python:3.12-slim
                WORKDIR /app
                COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
                COPY . .
                EXPOSE 8080
                CMD ["python", "main.py"]
            """),
        ),
        (
            "Rewrite the test file — it's outdated and doesn't match the current API",
            "I'll rewrite the test file to match the current API.",
            "tests/test_api.py",
            textwrap.dedent("""\
                import pytest
                from app import app


                @pytest.fixture
                def client():
                    app.config['TESTING'] = True
                    with app.test_client() as c:
                        yield c


                def test_index(client):
                    resp = client.get('/')
                    assert resp.status_code == 200


                def test_health(client):
                    resp = client.get('/health')
                    assert resp.status_code == 200
                    assert resp.json['status'] == 'ok'


                def test_not_found(client):
                    resp = client.get('/nonexistent')
                    assert resp.status_code == 404
            """),
        ),
        (
            "Rewrite utils.py from scratch — the current code is spaghetti",
            "I'll rewrite utils.py with clean, focused functions.",
            "utils.py",
            textwrap.dedent("""\
                import hashlib
                import re
                from datetime import datetime


                def slugify(text):
                    text = text.lower().strip()
                    text = re.sub(r'[^\\w\\s-]', '', text)
                    return re.sub(r'[-\\s]+', '-', text)


                def hash_string(s):
                    return hashlib.sha256(s.encode()).hexdigest()


                def format_timestamp(dt=None):
                    dt = dt or datetime.now()
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
            """),
        ),
        (
            "Start this file over from scratch",
            "I'll rewrite the file from scratch.",
            "app.py",
            textwrap.dedent("""\
                from flask import Flask

                app = Flask(__name__)


                @app.route('/')
                def index():
                    return {'status': 'ok'}


                if __name__ == '__main__':
                    app.run(port=8080)
            """),
        ),
        (
            "Rewrite server.py as a FastAPI app",
            "I'll rewrite server.py using FastAPI.",
            "server.py",
            textwrap.dedent("""\
                import uvicorn
                from fastapi import FastAPI

                app = FastAPI()


                @app.get('/')
                def root():
                    return {'message': 'ok'}


                if __name__ == '__main__':
                    uvicorn.run(app, host='0.0.0.0', port=8080)
            """),
        ),
        (
            "Rewrite the user model with SQLAlchemy",
            "I'll rewrite the user model using SQLAlchemy.",
            "models/user.py",
            textwrap.dedent("""\
                from sqlalchemy import Column, Integer, String, DateTime
                from sqlalchemy.sql import func
                from .base import Base


                class User(Base):
                    __tablename__ = 'users'

                    id = Column(Integer, primary_key=True)
                    name = Column(String(100), nullable=False)
                    email = Column(String(255), unique=True, nullable=False)
                    created_at = Column(DateTime, server_default=func.now())

                    def __repr__(self):
                        return f'<User {self.name}>'
            """),
        ),
    ]

    results = []
    verified = 0
    rejected = 0
    for user_msg, reasoning, path, content in items:
        if path.endswith(".py"):
            if not verify_python(content):
                print(f"  REJECTED (syntax error): {path} for '{user_msg[:50]}'")
                rejected += 1
                continue
            verified += 1
        results.append(example(
            msg("system", SYSTEM_PROMPT),
            msg("user", user_msg),
            msg("assistant", f"{reasoning}\n\n{tool_call('write_file', {'path': path, 'content': content})}"),
        ))
    print(f"  write_file: {verified} verified, {rejected} rejected")
    return results


# ---------------------------------------------------------------------------
# EDIT_FILE examples
# ---------------------------------------------------------------------------
def edit_file_examples():
    items = [
        ("Change the port from 8080 to 3000 in server.py", "I'll update the port number.",
         {"path": "server.py", "old_text": "port=8080", "new_text": "port=3000"}),
        ("Fix the typo in main.py — it says 'recieve' instead of 'receive'", "I'll fix that spelling mistake.",
         {"path": "main.py", "old_text": "recieve", "new_text": "receive"}),
        ("Rename the function calculate_total to compute_total in utils.py", "I'll rename that function.",
         {"path": "utils.py", "old_text": "def calculate_total(", "new_text": "def compute_total("}),
        ("Add error handling to the database connection in db.py", "I'll wrap the connection in a try/except.",
         {"path": "db.py", "old_text": "conn = sqlite3.connect(DB_PATH)\nreturn conn",
          "new_text": "try:\n    conn = sqlite3.connect(DB_PATH)\n    return conn\nexcept sqlite3.Error as e:\n    print(f'Database error: {e}')\n    raise"}),
        ("Change the import from flask to fastapi in app.py", "I'll update the import statement.",
         {"path": "app.py", "old_text": "from flask import Flask", "new_text": "from fastapi import FastAPI"}),
        ("Update the version number to 2.0.0 in setup.py", "I'll bump the version.",
         {"path": "setup.py", "old_text": "version='1.0.0'", "new_text": "version='2.0.0'"}),
        ("In config.py, change DEBUG = True to DEBUG = False", "I'll disable debug mode.",
         {"path": "config.py", "old_text": "DEBUG = True", "new_text": "DEBUG = False"}),
        ("Add a return type hint to get_user in models/user.py", "I'll add the return type annotation.",
         {"path": "models/user.py", "old_text": "def get_user(user_id):", "new_text": "def get_user(user_id) -> User | None:"}),
        ("Replace print statements with logging in worker.py", "I'll switch from print to proper logging.",
         {"path": "worker.py", "old_text": "print(f'Processing task {task_id}')", "new_text": "logger.info(f'Processing task {task_id}')"}),
        ("Change the max retries from 3 to 5 in the API client", "I'll increase the retry count.",
         {"path": "api_client.py", "old_text": "MAX_RETRIES = 3", "new_text": "MAX_RETRIES = 5"}),
        ("Fix the off-by-one error in the pagination", "I'll fix the pagination offset calculation.",
         {"path": "views.py", "old_text": "offset = page * PAGE_SIZE", "new_text": "offset = (page - 1) * PAGE_SIZE"}),
        ("Add requests to requirements.txt", "I'll add the requests dependency.",
         {"path": "requirements.txt", "old_text": "flask>=3.0", "new_text": "flask>=3.0\nrequests>=2.31"}),
        ("Change the database URL to use PostgreSQL instead of SQLite", "I'll update the database connection string.",
         {"path": "config.py", "old_text": "DATABASE_URL = 'sqlite:///app.db'", "new_text": "DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://localhost/myapp')"}),
        ("Fix the SQL injection vulnerability in the search function", "I'll switch to parameterized queries.",
         {"path": "search.py", "old_text": "cursor.execute(f\"SELECT * FROM items WHERE name LIKE '%{query}%'\")",
          "new_text": "cursor.execute('SELECT * FROM items WHERE name LIKE ?', (f'%{query}%',))"}),
        ("Add a cache decorator to get_settings in config.py", "I'll add caching with lru_cache.",
         {"path": "config.py", "old_text": "def get_settings():", "new_text": "@lru_cache(maxsize=1)\ndef get_settings():"}),
        ("Update the test assertion to check for status 201 not 200", "I'll fix the expected status code.",
         {"path": "tests/test_api.py", "old_text": "assert response.status_code == 200", "new_text": "assert response.status_code == 201"}),
        ("Add a logout route to app.py", "I'll add a logout route.",
         {"path": "app.py", "old_text": "if __name__ == '__main__':",
          "new_text": "@app.route('/logout')\ndef logout():\n    session.clear()\n    return redirect('/')\n\n\nif __name__ == '__main__':"}),
        ("Add CORS support to the Flask app", "I'll add CORS configuration.",
         {"path": "app.py", "old_text": "app = Flask(__name__)", "new_text": "app = Flask(__name__)\nCORS(app)"}),
        ("Change the log level from DEBUG to INFO in production config", "I'll update the log level.",
         {"path": "config/production.py", "old_text": "LOG_LEVEL = 'DEBUG'", "new_text": "LOG_LEVEL = 'INFO'"}),
        ("Make the timeout configurable via environment variable", "I'll read the timeout from an env var.",
         {"path": "client.py", "old_text": "TIMEOUT = 30", "new_text": "TIMEOUT = int(os.environ.get('CLIENT_TIMEOUT', '30'))"}),
    ]
    return [
        example(
            msg("system", SYSTEM_PROMPT),
            msg("user", u),
            msg("assistant", f"{r}\n\n{tool_call('edit_file', p)}"),
        )
        for u, r, p in items
    ]


# ---------------------------------------------------------------------------
# GREP_CODEBASE examples
# ---------------------------------------------------------------------------
def grep_codebase_examples():
    items = [
        ("Where is the login function defined?", "I'll search for the login function definition.",
         {"pattern": "def login", "directory": "."}),
        ("Find all TODO comments in the code", "I'll search for TODO comments across the project.",
         {"pattern": "TODO", "directory": "."}),
        ("Where do we import requests?", "I'll find all files that import requests.",
         {"pattern": "import requests", "directory": "."}),
        ("Find everywhere we use get_user_by_id", "I'll find all usages of that function.",
         {"pattern": "get_user_by_id", "directory": "."}),
        ("Where is DATABASE_URL set?", "I'll search for where DATABASE_URL is defined.",
         {"pattern": "DATABASE_URL", "directory": "."}),
        ("Find all API endpoints in the project", "I'll search for route decorators.",
         {"pattern": "@app.route", "directory": "."}),
        ("Look for any print statements in src/", "I'll search for print statements in src.",
         {"pattern": "print(", "directory": "src"}),
        ("Where do we handle the 404 error?", "I'll search for 404 error handling.",
         {"pattern": "404", "directory": "."}),
        ("Search for uses of os.environ in the config directory", "I'll find environment variable usage.",
         {"pattern": "os.environ", "directory": "config"}),
        ("Where is the send_email function called?", "I'll find all call sites for send_email.",
         {"pattern": "send_email(", "directory": "."}),
        ("Show me where we raise ValueError", "I'll find all places that raise ValueError.",
         {"pattern": "raise ValueError", "directory": "."}),
        ("Find all async functions in the project", "I'll search for async function definitions.",
         {"pattern": "async def ", "directory": "."}),
        ("Where is MAX_RETRIES used?", "I'll find all references to MAX_RETRIES.",
         {"pattern": "MAX_RETRIES", "directory": "."}),
        ("Search for any use of eval()", "I'll search for eval usage.",
         {"pattern": "eval(", "directory": "."}),
        ("where tf is the login function", "I'll search for the login function.",
         {"pattern": "def login", "directory": "."}),
        ("Can you locate where login is defined?", "I'll search for the login function definition.",
         {"pattern": "def login", "directory": "."}),
        ("find login", "I'll search for the login function.",
         {"pattern": "def login", "directory": "."}),
        ("Search for the login function", "I'll search for the login function definition.",
         {"pattern": "def login", "directory": "."}),
        ("I need to find the login implementation", "I'll search for the login function.",
         {"pattern": "def login", "directory": "."}),
        ("Find where the database connection is configured", "I'll search for database connection configuration.",
         {"pattern": "database", "directory": "."}),
    ]
    return [
        example(
            msg("system", SYSTEM_PROMPT),
            msg("user", u),
            msg("assistant", f"{r}\n\n{tool_call('grep_codebase', p)}"),
        )
        for u, r, p in items
    ]


# ---------------------------------------------------------------------------
# LIST_FILES examples
# ---------------------------------------------------------------------------
def list_files_examples():
    items = [
        ("What files are in this project?", "I'll show you the project structure.", "."),
        ("Show me the directory structure", "I'll list the file tree.", "."),
        ("What's inside the src folder?", "I'll list the contents of the src directory.", "src"),
        ("List all the test files", "I'll show what's in the tests directory.", "tests"),
        ("What does the project structure look like?", "I'll display the file tree.", "."),
        ("Show me what's in the config directory", "I'll list the config directory contents.", "config"),
        ("What files exist in app/models?", "I'll check what model files exist.", "app/models"),
        ("ls", "I'll list the files in the current directory.", "."),
        ("tree", "I'll show the full project tree.", "."),
        ("What's in the scripts folder?", "I'll list the scripts directory.", "scripts"),
        ("Show me all files under lib/", "I'll list everything in the lib directory.", "lib"),
        ("What components exist in src/components?", "I'll list the components directory.", "src/components"),
        ("Show me the project layout", "I'll display the project's file structure.", "."),
        ("What static assets do we have?", "I'll check the static files directory.", "static"),
        ("List files in the deployment config", "I'll show the deployment directory contents.", "deploy"),
        ("What templates do we have?", "I'll check the templates directory.", "templates"),
        ("Show me how the project is organized", "I'll display the project structure.", "."),
        ("Do we have any files in the docs directory?", "I'll check what's in docs.", "docs"),
        ("What migration files do we have?", "I'll list the migrations directory.", "db/migrations"),
        ("Find what files are in the project", "I'll list the project files.", "."),
    ]
    return [
        example(
            msg("system", SYSTEM_PROMPT),
            msg("user", u),
            msg("assistant", f"{r}\n\n{tool_call('list_files', {'directory': d})}"),
        )
        for u, r, d in items
    ]


# ---------------------------------------------------------------------------
# RUN_COMMAND examples
# ---------------------------------------------------------------------------
def run_command_examples():
    items = [
        ("Run the tests", "I'll run the test suite.", "pytest tests/ -v"),
        ("Check if the code has any linting errors", "I'll run the linter.", "ruff check ."),
        ("What Python version am I using?", "I'll check the Python version.", "python3 --version"),
        ("Install the requirements", "I'll install the project dependencies.", "pip install -r requirements.txt"),
        ("Run the main script", "I'll execute the main script.", "python3 main.py"),
        ("Check if port 8080 is in use", "I'll check what's listening on that port.", "lsof -i :8080"),
        ("Format the code with ruff", "I'll run the code formatter.", "ruff format ."),
        ("Show me the git log", "I'll display the recent git history.", "git log --oneline -10"),
        ("What branch am I on?", "I'll check the current git branch.", "git branch --show-current"),
        ("Run just the test for the user model", "I'll run that specific test file.", "pytest tests/test_user.py -v"),
        ("How big is the project directory?", "I'll check the directory size.", "du -sh ."),
        ("Start the dev server", "I'll start the development server.", "python3 app.py"),
        ("Run mypy type checking", "I'll run the type checker.", "mypy src/ --ignore-missing-imports"),
        ("Check if git is clean", "I'll check the git status.", "git status"),
        ("Count lines of Python code in the project", "I'll count the lines of Python code.",
         "find . -name '*.py' -not -path './.venv/*' | xargs wc -l"),
        ("Check what packages are installed", "I'll list the installed packages.", "pip list"),
        ("Build the Docker image", "I'll build the Docker image.", "docker build -t myapp ."),
        ("Run this Python file", "I'll run it.", "python3 main.py"),
        ("Check if the tests pass", "I'll run the test suite.", "pytest tests/ -v"),
        ("Verify that the code compiles", "I'll check for syntax errors.", "python3 -m py_compile main.py"),
    ]
    return [
        example(
            msg("system", SYSTEM_PROMPT),
            msg("user", u),
            msg("assistant", f"{r}\n\n{tool_call('run_command', {'command': c})}"),
        )
        for u, r, c in items
    ]


# ---------------------------------------------------------------------------
# TOOL SELECTION examples — ambiguous phrasing, correct tool
# ---------------------------------------------------------------------------
def tool_selection_examples():
    items = [
        ("Find where the database connection is configured", "I'll search the codebase for database configuration.",
         "grep_codebase", {"pattern": "database", "directory": "."}),
        ("Find what files are in the project", "I'll list the project files.",
         "list_files", {"directory": "."}),
        ("Check what's in the config file", "I'll read the config file.",
         "read_file", {"path": "config.py"}),
        ("Check if the tests pass", "I'll run the test suite.",
         "run_command", {"command": "pytest tests/ -v"}),
        ("Look at the error handling in server.py", "I'll read the server file.",
         "read_file", {"path": "server.py"}),
        ("Look for any uses of the deprecated API", "I'll search for deprecated API references.",
         "grep_codebase", {"pattern": "deprecated", "directory": "."}),
        ("Show me the main entry point", "I'll read the main file.",
         "read_file", {"path": "main.py"}),
        ("Show me how the project is organized", "I'll display the project structure.",
         "list_files", {"directory": "."}),
        ("Make a utility function that formats dates", "I'll create a utility file.",
         "create_file", {"path": "utils/dates.py", "content": "from datetime import datetime\n\n\ndef format_date(dt, fmt='%Y-%m-%d'):\n    return dt.strftime(fmt)\n\n\ndef parse_date(s, fmt='%Y-%m-%d'):\n    return datetime.strptime(s, fmt)\n"}),
        ("Fix the bug — it should split on comma not space", "I'll fix the split delimiter.",
         "edit_file", {"path": "parser.py", "old_text": "line.split(' ')", "new_text": "line.split(',')"}),
        ("Verify that the code compiles", "I'll check for syntax errors.",
         "run_command", {"command": "python3 -m py_compile main.py"}),
        ("Where do we validate email addresses?", "I'll search for email validation code.",
         "grep_codebase", {"pattern": "email", "directory": "."}),
        ("Rewrite the entire server.py", "The file needs a full rewrite.",
         "write_file", {"path": "server.py", "content": "from flask import Flask\n\napp = Flask(__name__)\n\n\n@app.route('/')\ndef index():\n    return {'status': 'ok'}\n\n\nif __name__ == '__main__':\n    app.run(port=8080)\n"}),
        ("Change the port in server.py from 8080 to 3000", "That's a small targeted change.",
         "edit_file", {"path": "server.py", "old_text": "port=8080", "new_text": "port=3000"}),
        ("Create a brand new helper module", "This is a new file, I'll create it.",
         "create_file", {"path": "lib/helpers.py", "content": "def clamp(value, low, high):\n    return max(low, min(value, high))\n"}),
        ("Overwrite the Makefile with just test and run targets", "I'll overwrite it.",
         "write_file", {"path": "Makefile", "content": ".PHONY: test run\n\ntest:\n\tpytest tests/ -v\n\nrun:\n\tpython3 main.py\n"}),
    ]
    return [
        example(
            msg("system", SYSTEM_PROMPT),
            msg("user", u),
            msg("assistant", f"{r}\n\n{tool_call(t, p)}"),
        )
        for u, r, t, p in items
    ]


# ---------------------------------------------------------------------------
# POST-TOOL SUMMARIZATION — model sees tool result and responds cleanly
# ---------------------------------------------------------------------------
def post_tool_summary_examples():
    results = []

    # read_file → summarize
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "What's in main.py?"),
        msg("assistant", f"Let me read that file.\n\n{tool_call('read_file', {'path': 'main.py'})}"),
        msg("user", "[Tool result] File: main.py (10 lines)\n   1 | import sys\n   2 | \n   3 | def main():\n   4 |     print('Hello')\n   5 | \n   6 | if __name__ == '__main__':\n   7 |     main()\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "main.py is a simple script with a `main()` function that prints 'Hello'. It uses the standard `if __name__ == '__main__'` pattern to run when executed directly."),
    ))

    # create_file → confirm
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Create a hello world script"),
        msg("assistant", f"I'll create a hello world script.\n\n{tool_call('create_file', {'path': 'hello.py', 'content': 'print(\"Hello, world!\")\\n'})}"),
        msg("user", "[Tool result] Created hello.py (25 bytes)\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "Done! I created `hello.py` with a simple hello world print statement. You can run it with `python3 hello.py`."),
    ))

    # list_files → summarize structure
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "What files are in this project?"),
        msg("assistant", f"I'll show you the project structure.\n\n{tool_call('list_files', {'directory': '.'})}"),
        msg("user", "[Tool result] ./\n  main.py\n  utils.py\n  config.py\n  tests/\n    test_main.py\n    test_utils.py\n  requirements.txt\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "The project has three main Python files (`main.py`, `utils.py`, `config.py`), a `tests/` directory with two test files, and a `requirements.txt`."),
    ))

    # grep → summarize findings
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Where is the login function?"),
        msg("assistant", f"I'll search for the login function.\n\n{tool_call('grep_codebase', {'pattern': 'def login', 'directory': '.'})}"),
        msg("user", "[Tool result] Found 2 matches:\nauth/views.py:15 | def login(username, password):\nauth/views.py:42 | def login_required(f):\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "Found it! The `login` function is defined at `auth/views.py` line 15. There's also a `login_required` decorator at line 42 in the same file."),
    ))

    # edit_file → confirm change
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Change the port from 8080 to 3000"),
        msg("assistant", f"I'll update the port number.\n\n{tool_call('edit_file', {'path': 'server.py', 'old_text': 'port=8080', 'new_text': 'port=3000'})}"),
        msg("user", "[Tool result] Successfully edited server.py\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "Done! Changed the port from 8080 to 3000 in server.py."),
    ))

    # run_command → summarize output
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Run the tests"),
        msg("assistant", f"I'll run the test suite.\n\n{tool_call('run_command', {'command': 'pytest tests/ -v'})}"),
        msg("user", "[Tool result] ===== test session starts =====\ntests/test_main.py::test_hello PASSED\ntests/test_utils.py::test_add PASSED\ntests/test_utils.py::test_subtract PASSED\n===== 3 passed in 0.05s =====\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "All 3 tests passed! One test in test_main.py and two in test_utils.py — everything looks good."),
    ))

    # write_file → confirm rewrite
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Rewrite config.py to use dataclasses"),
        msg("assistant", f"I'll rewrite config.py using a dataclass.\n\n{tool_call('write_file', {'path': 'config.py', 'content': 'from dataclasses import dataclass\\n\\n@dataclass\\nclass Config:\\n    debug: bool = False\\n    port: int = 8080\\n'})}"),
        msg("user", "[Tool result] Wrote config.py (89 bytes)\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "Done! I rewrote config.py to use a `Config` dataclass with `debug` and `port` fields. Much cleaner than the old version."),
    ))

    # run_command with error → summarize error
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Run main.py"),
        msg("assistant", f"I'll run it.\n\n{tool_call('run_command', {'command': 'python3 main.py'})}"),
        msg("user", "[Tool result] stderr: Traceback (most recent call last):\n  File \"main.py\", line 1, in <module>\n    import flask\nModuleNotFoundError: No module named 'flask'\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "main.py failed because Flask isn't installed. Run `pip install flask` first, or `pip install -r requirements.txt` if you have a requirements file."),
    ))

    # read_file on larger file → concise summary
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "What's in utils.py?"),
        msg("assistant", f"Let me read that file.\n\n{tool_call('read_file', {'path': 'utils.py'})}"),
        msg("user", "[Tool result] File: utils.py (45 lines)\n   1 | import hashlib\n   2 | import re\n   3 | from datetime import datetime\n   4 | \n   5 | def slugify(text):\n   6 |     text = text.lower().strip()\n   7 |     text = re.sub(r'[^\\w\\s-]', '', text)\n   8 |     return re.sub(r'[-\\s]+', '-', text)\n   9 | \n  10 | def hash_string(s):\n  11 |     return hashlib.sha256(s.encode()).hexdigest()\n  12 | \n  13 | def format_timestamp(dt=None):\n  14 |     dt = dt or datetime.now()\n  15 |     return dt.strftime('%Y-%m-%d %H:%M:%S')\n\n[... truncated]\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "utils.py has three utility functions: `slugify()` for converting text to URL-safe slugs, `hash_string()` for SHA-256 hashing, and `format_timestamp()` for formatting dates."),
    ))

    # grep with no results → clean response
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Find any uses of eval()"),
        msg("assistant", f"I'll search for eval usage.\n\n{tool_call('grep_codebase', {'pattern': 'eval(', 'directory': '.'})}"),
        msg("user", "[Tool result] No matches found.\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "Good news — no uses of `eval()` found in the project. The codebase looks clean on that front."),
    ))

    return results


# ---------------------------------------------------------------------------
# ERROR RECOVERY — tool fails, model tries a different approach
# ---------------------------------------------------------------------------
def error_recovery_examples():
    results = []

    # create_file fails (exists) → use write_file
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Create a config.py with default settings"),
        msg("assistant", f"I'll create the config file.\n\n{tool_call('create_file', {'path': 'config.py', 'content': 'DEBUG = False\\nPORT = 8080\\n'})}"),
        msg("user", "[Tool result] Error: File config.py already exists\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", f"That file already exists. I'll overwrite it instead.\n\n{tool_call('write_file', {'path': 'config.py', 'content': 'DEBUG = False\\nPORT = 8080\\n'})}"),
    ))

    # edit_file fails (text not found) → read first
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Change the port to 3000 in server.py"),
        msg("assistant", f"I'll update the port.\n\n{tool_call('edit_file', {'path': 'server.py', 'old_text': 'port=8080', 'new_text': 'port=3000'})}"),
        msg("user", "[Tool result] Error: old_text not found in server.py\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", f"The exact text wasn't found. Let me read the file to see how the port is currently set.\n\n{tool_call('read_file', {'path': 'server.py'})}"),
    ))

    # run_command fails → suggest fix
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Run the tests"),
        msg("assistant", f"I'll run the test suite.\n\n{tool_call('run_command', {'command': 'pytest tests/ -v'})}"),
        msg("user", "[Tool result] stderr: bash: pytest: command not found\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", f"pytest isn't installed. Let me install it first.\n\n{tool_call('run_command', {'command': 'pip install pytest'})}"),
    ))

    # read_file fails (not found) → list files to find it
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Show me the main file"),
        msg("assistant", f"I'll read the main file.\n\n{tool_call('read_file', {'path': 'main.py'})}"),
        msg("user", "[Tool result] Error: File not found: main.py\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", f"main.py doesn't exist. Let me check what files are in the project.\n\n{tool_call('list_files', {'directory': '.'})}"),
    ))

    # Permission error on run_command
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Run the deploy script"),
        msg("assistant", f"I'll run the deploy script.\n\n{tool_call('run_command', {'command': 'bash scripts/deploy.sh'})}"),
        msg("user", "[Tool result] stderr: bash: scripts/deploy.sh: Permission denied\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", f"The script doesn't have execute permission. Let me fix that.\n\n{tool_call('run_command', {'command': 'chmod +x scripts/deploy.sh'})}"),
    ))

    return results


# ---------------------------------------------------------------------------
# MULTI-STEP examples (read → edit, grep → read, create → run)
# ---------------------------------------------------------------------------
def multi_step_examples():
    results = []

    # Read then edit
    health_new_text = "@app.route('/health')\ndef health():\n    return jsonify({'status': 'ok'})\n\n\nif __name__ == '__main__':"
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Add a health check endpoint to app.py"),
        msg("assistant", "I'll first read app.py to see the current code.\n\n" + tool_call('read_file', {'path': 'app.py'})),
        msg("user", "[Tool result] File: app.py (10 lines)\n   1 | from flask import Flask, jsonify\n   2 | \n   3 | app = Flask(__name__)\n   4 | \n   5 | @app.route('/')\n   6 | def index():\n   7 |     return jsonify({'message': 'hello'})\n   8 | \n   9 | if __name__ == '__main__':\n  10 |     app.run(debug=True)"),
        msg("assistant", "I can see the app structure. I'll add a health check endpoint before the main block.\n\n" + tool_call('edit_file', {'path': 'app.py', 'old_text': "if __name__ == '__main__':", 'new_text': health_new_text})),
    ))

    # Grep then read
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "I want to understand how authentication works in this project"),
        msg("assistant", "I'll search for authentication-related code first.\n\n" + tool_call('grep_codebase', {'pattern': 'auth', 'directory': '.'})),
        msg("user", "[Tool result] Found 4 matches:\nmiddleware/auth.py:5 | def check_auth(request):\nmiddleware/auth.py:12 | def verify_token(token):\nroutes/login.py:8 | def login(username, password):\nconfig.py:3 | AUTH_SECRET = os.environ.get('AUTH_SECRET')"),
        msg("assistant", "I found the auth middleware. Let me read it to understand the full implementation.\n\n" + tool_call('read_file', {'path': 'middleware/auth.py'})),
    ))

    # List then read
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "What tests do we have? Show me the most important one."),
        msg("assistant", "I'll check what test files exist.\n\n" + tool_call('list_files', {'directory': 'tests'})),
        msg("user", "[Tool result] tests/\n  conftest.py\n  test_api.py\n  test_auth.py\n  test_models.py"),
        msg("assistant", "The auth tests are likely the most critical. Let me read those.\n\n" + tool_call('read_file', {'path': 'tests/test_auth.py'})),
    ))

    # Create then run
    test_content = "from math_utils import add\n\ndef test_add_positive():\n    assert add(2, 3) == 5\n\ndef test_add_negative():\n    assert add(-1, -1) == -2\n\ndef test_add_zero():\n    assert add(0, 0) == 0\n"
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Write a test for the add function and run it"),
        msg("assistant", "I'll create the test file first.\n\n" + tool_call('create_file', {'path': 'tests/test_math.py', 'content': test_content})),
        msg("user", "[Tool result] Created tests/test_math.py (150 bytes)\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "Test file created. Now I'll run it.\n\n" + tool_call('run_command', {'command': 'pytest tests/test_math.py -v'})),
    ))

    # Edit then verify
    results.append(example(
        msg("system", SYSTEM_PROMPT),
        msg("user", "Change the default timeout to 60 seconds and make sure the tests still pass"),
        msg("assistant", "I'll update the timeout value.\n\n" + tool_call('edit_file', {'path': 'config.py', 'old_text': 'TIMEOUT = 30', 'new_text': 'TIMEOUT = 60'})),
        msg("user", "[Tool result] Successfully edited config.py\n\nNow respond to the user. Summarize what you found or did. Do NOT call the same tool again."),
        msg("assistant", "Timeout updated. Let me run the tests to make sure nothing broke.\n\n" + tool_call('run_command', {'command': 'pytest tests/ -v'})),
    ))

    return results


# ---------------------------------------------------------------------------
# MAIN — generate, verify, and save
# ---------------------------------------------------------------------------
def main():
    all_examples = []

    generators = [
        ("no_tool", no_tool_examples),
        ("read_file", read_file_examples),
        ("create_file", create_file_examples),
        ("write_file", write_file_examples),
        ("edit_file", edit_file_examples),
        ("grep_codebase", grep_codebase_examples),
        ("list_files", list_files_examples),
        ("run_command", run_command_examples),
        ("tool_selection", tool_selection_examples),
        ("post_tool_summary", post_tool_summary_examples),
        ("error_recovery", error_recovery_examples),
        ("multi_step", multi_step_examples),
    ]

    for name, gen_fn in generators:
        examples = gen_fn()
        print(f"  {name}: {len(examples)} examples")
        all_examples.extend(examples)

    random.seed(42)
    random.shuffle(all_examples)

    split = int(len(all_examples) * 0.9)
    train = all_examples[:split]
    valid = all_examples[split:]

    out_dir = os.path.join(os.path.dirname(__file__), "training_data")
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, "train.jsonl")
    valid_path = os.path.join(out_dir, "valid.jsonl")

    with open(train_path, "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    with open(valid_path, "w") as f:
        for ex in valid:
            f.write(json.dumps(ex) + "\n")

    print(f"\nTotal: {len(all_examples)} examples")
    print(f"Train: {len(train)} -> {train_path}")
    print(f"Valid: {len(valid)} -> {valid_path}")

    # Validate all examples
    import re
    errors = 0
    for i, ex in enumerate(all_examples):
        msgs = ex["messages"]
        if msgs[0]["role"] != "system":
            print(f"  ERROR example {i}: missing system message")
            errors += 1
        assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
        for am in assistant_msgs:
            if "[TOOL_CALL]" in am["content"]:
                tc_match = re.search(r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]", am["content"], re.DOTALL)
                if tc_match:
                    try:
                        tc = json.loads(tc_match.group(1))
                        if "name" not in tc or "parameters" not in tc:
                            print(f"  ERROR example {i}: tool call missing name or parameters")
                            errors += 1
                    except json.JSONDecodeError:
                        print(f"  ERROR example {i}: invalid JSON in tool call")
                        errors += 1

    if errors == 0:
        print("\nAll examples validated successfully.")
    else:
        print(f"\n{errors} errors found!")


if __name__ == "__main__":
    main()
