Tool_level_prompt='''You are a research assistant. Please generate a coarse-grained tool usage instruction based on detailed user instructions for a tool usage task. \
You should not provide a detailed task description and need to include the tool name in the simplified instruction.

Example1: 
System: I'm planning a surprise party for my best friend's birthday.
Tool: The Cocktail DB, Weather, Free NBA.
Answer: I'm planning a surprise party for my best friend's birthday. Using The Cocktail DB, Weather and Free NBA to find me some cocktial recipe, weather forecast and basketball information.


Example2: 
System: I'm organizing a charity event for my company and we need some assistance.
Tool: Microsoft Translator Text, MyMemory - Translation Memory.
Answer: I'm organizing a charity event for my company and we need some assistance. Using these two tools, Microsoft Translator Text, MyMemory - Translation Memory, and give me some ideas.

Now, Please make the simplified answer of below requests.

System: {request}
Answer:'''


API_level_prompt='''You are a research assistant. Please generate a coarse-grained tool usage instruction based on detailed user instructions for a tool usage task. \
You should not provide a detailed task description and need to include the api name in the simplified instruction.

Example1: 
System: I'm planning a surprise party for my best friend's birthday.
API: Detailed Cocktail Recipe by ID, 16 Day Forecast, Get a Specific Game.
Answer: I'm planning a surprise party for my best friend's birthday. Using Detailed Cocktail Recipe by ID, 16 Day Forecast, Get a Specific Game to find me some cocktial recipe, weather forecast and basketball information.


Example2: 
System: I'm organizing a charity event for my company and we need some assistance.
API: Languages, search translations.
Answer: I'm organizing a charity event for my company and we need some assistance. Using these two APIs,  Languages, search translations, and give me some ideas.

Now, Please make the simplified answer of below requests.

System: {request}
Answer:'''


Category_level_prompt='''You are a research assistant. Please generate a coarse-grained tool usage instruction based on detailed user instructions for a tool usage task. \
You should not provide a detailed task description and need to include the api name in the simplified instruction.

Example1: 
System: I'm planning a surprise party for my best friend's birthday.
Category: Food, Weather, Sports.
Answer: I'm planning a surprise party for my best friend's birthday. Please help me find some information with tools from Food, Weather, Sports categories.


Example2: 
System: I'm organizing a charity event for my company and we need some assistance.
Category: Translation, Business.
Answer: I'm organizing a charity event for my company and we need some assistance. Using tools from Translation and Business category, and give me some ideas.

Now, Please make the simplified answer of below requests.

System: {request}
Answer:'''
