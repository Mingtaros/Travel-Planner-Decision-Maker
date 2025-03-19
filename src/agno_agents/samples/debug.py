# import inspect
# from agno.vectordb.lancedb import LanceDb

# print(inspect.signature(LanceDb.__init__))

# from agno.vectordb.lancedb import SearchType
# help(SearchType)


from agno.utils.query_expansion import expand_query

query = "recommend me food places as I have a sweet tooth."
expanded_query = expand_query(query)
print(f"🔍 Expanded Query: {expanded_query}")

response = agent.run(expanded_query, stream=False)