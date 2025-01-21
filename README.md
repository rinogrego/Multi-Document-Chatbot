# CDT-Internal-Chatbot
LLM Chatbot to converse with internal project documentation

## Features

- Notion integration
- OCR for images within a PDF
- History-aware conversation
- Project-based chat management

## Flow Process

- Home page -> Create or Choose a Project to work on
- When selecting a Project:
  - initiate an empty vectorstore
  - as the knowledge base increased, update the vectorstore
- After selecting a Project:
  - `Knowledge Base`: displays all knowledge stored for that Project
  - `Knowledge Update`: the save goes to storage for that Project
  - `Chat Room`: display a list of conversation related to that Project below the Chat Room, and when clicked it will display that conversation. 
    - to change convo, there is a button below the 4 Tabs
    - display the list of conversation like displaying the 4 Tabs with scrollable choice
  - `Chat History`: show what is currently active conversation