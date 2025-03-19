## Checking of Agno Agent Class parameters
#  from inspect import signature
# from agno.agent import Agent

# # Get the parameters of the Agent class
# agent_signature = signature(Agent.__init__)

# print("Agent class accepts the following arguments:\n")
# for param in agent_signature.parameters.values():
#     print(f"- {param.name}: {param.annotation} (default: {param.default})")

#================================================

# ## Checking of chromadb collection exists or not 
# from agno.vectordb.chroma import ChromaDb

# # Initialize ChromaDB connection
# db = ChromaDb(collection="recipes")

# ## Check if the "recipes" collection exists
# try:
#     client = db.client  # Access underlying ChromaDB client
#     print("Existing collections:", client.list_collections())
# except AttributeError:
#     print("Error: Could not access ChromaDB client. Check if the database is initialized properly.")

# from inspect import signature
# from agno.vectordb.chroma import ChromaDb

# # Print the constructor's parameter list
# print(signature(ChromaDb.__init__))
