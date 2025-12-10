# Rollout plan

1. ~~Complete project with **single LLM call**.~~ (Done)
    - Brief descriptions about the project, model and the data along with a few explainability results are curated and given to the model. The model then provides the answer. Aim is to complete v1 of all components (frontend and backend).

2. ReACT agent with explainability tools
    - LLM makes tool calls, and refines it's response.

3. Multi-LLm actor-critic 
    - 1 LLM gets explainability, writes response
    - The other critcs the response based on guidelines+response-requirements.

## items

- figure out data and understand the proposed bias.
- figure out explainability technique to express the bias
  - Shaply values
  - other relevant explainability techniques
- ~~find guidelines that explicitely cautions about this particular bias~~
- ~~promptify guidelines~~
- toolify a few explainability techniques
- show that the agent calls appropriate explainability techniques
- show that the agent is able to specifity which guidelines are violated
    and generates a user-ready explanation displaying the bias.
