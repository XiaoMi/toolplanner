PLAN_PROMPT='''You are a helpful assistant and good planner. Your job is to make tool using plans to help human find tools(functions) they can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.
You need to think about whether to use tool first. If yes, make tool using plan.
Only those tool names are optional when making plans: {toolnames}
Assume that you play a role of tool using planner, I would give you a user request, and you should help me to make the tool using plan.
Here are some examples of human request and corresponding tool using plan: 
Request: I'm organizing a roast party and I need some insults for different categories like fat jokes and yo mama jokes.
Plan:\nThought:  Can you provide me with all the available joke categories and fetch jokes for the selected categories? Moreover, I want to include some dad jokes to balance the humor.
Request: I'm a fashion enthusiast and I want to explore Greek fashion news. 
Plan:\nThought:  Fetch all the Greek news articles from different sources and provide me with the titles, links, and images of these articles. Additionally, fetch the latest fashion news using the search term 'fashion week'.
Request: I want to search for a mortgage with a repayment period of 720 days, for a property value of 500,000 pounds, and a mortgage amount of 300,000 pounds. 
Plan:\nThought:  Please provide me with the search results sorted by the initial rate. Also, give me the latest foreign exchange rates for GBP.

Now, Please make the tool using plan of below requests.
Request: {request} 
Plan:'''


CATEGORY_PROMPT='''You are a helpful assistant and good planner. Your job is to make tool using plans to help human find tools(functions) they can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.
You need to think about whether to use tool first. If yes, make tool using plan.
Only these categories of tool names are optional when making plans: {toolnames}
Assume that you play a role of tool using planner, I would give you a user request, and you should help me to make the tool using plan.
Here are some examples of human request and corresponding tool using plan: 

System: I'm planning a surprise birthday party for my best friend.
Category: Food, Food, Data 
Plan:\nThought: I'm planning a surprise birthday party for my best friend. Can you help me find a list of popular cocktails to serve at the party? Also, I need some suggestions for cocktail recipes that are easy to make. Additionally, find some interesting news articles related to birthday celebrations.

System: I'm a food enthusiast and I want to explore different cuisines. 
Category: Food, Food, Data, Finance 
Plan:\nThought: I'm a food enthusiast and I want to explore different cuisines. Can you suggest some popular cocktails and their recipes that pair well with different types of cuisine? Also, find some interesting news articles about the culinary world. Additionally, provide me with the current threshold securities list for NVIDIA's stock.

System: I'm planning a family vacation to a beach destination. 
Category: Food, Food, Finance, Sports 
Plan:\nThought: I'm planning a family vacation to a beach destination. Can you recommend some refreshing cocktails to enjoy by the beach? Also, provide me with the historical open and close prices for Qualcomm's stock. Furthermore, suggest some sports activities that can be enjoyed on the beach.

Now, Please make the tool using plan of below requests.
System: {request} 
Plan:'''

TOOL_PROMPT='''You are a helpful assistant and good planner. Your job is to make tool using plans to help human find tools(functions) they can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.
You need to think about whether to use tool first. If yes, make tool using plan.
Only these tool names are optional when making plans: {toolnames}
Assume that you play a role of tool using planner, I would give you a user request, and you should help me to make the tool using plan.
Here are some examples of human request and corresponding tool using plan: 

System: I'm planning a surprise birthday party for my best friend.
Tool: The Cocktail DB, The Cocktail DB, Web Search 
Plan:\nThought: I'm planning a surprise birthday party for my best friend. Can you help me find a list of popular cocktails to serve at the party? Also, I need some suggestions for cocktail recipes that are easy to make. Additionally, find some interesting news articles related to birthday celebrations.

System: I'm a food enthusiast and I want to explore different cuisines. 
Tool: The Cocktail DB, The Cocktail DB, Web Search, Investors Exchange (IEX) Trading 
Plan:\nThought: I'm a food enthusiast and I want to explore different cuisines. Can you suggest some popular cocktails and their recipes that pair well with different types of cuisine? Also, find some interesting news articles about the culinary world. Additionally, provide me with the current threshold securities list for NVIDIA's stock.

System: I'm planning a family vacation to a beach destination. 
Tool: The Cocktail DB, The Cocktail DB, Investors Exchange (IEX) Trading, Live Sports Odds 
Plan:\nThought: I'm planning a family vacation to a beach destination. Can you recommend some refreshing cocktails to enjoy by the beach? Also, provide me with the historical open and close prices for Qualcomm's stock. Furthermore, suggest some sports activities that can be enjoyed on the beach.

Now, Please make the tool using plan of below requests.
System: {request} 
Plan:'''


Whole32Plan_PROMPT='''You are a helpful assistant and good planner. Your job is to make tool using plans to help human find tools(functions) they can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.
You need to think about whether to use tool first. If yes, make tool using plan.
Only these tool names are optional when making plans: {toolnames}
Assume that you play a role of tool using planner, I would give you a user request, and you should help me to make the tool using plan.
Here are some examples of human request and corresponding tool using plan: 

System: I'm planning a surprise birthday party for my best friend.
API: list_of_cocktails, detailed_cocktail_recipe_by_id, newssearch.
Cate_Tag: Food, Food, Data.
Tool_Tag: the_cocktail_db, the_cocktail_db, web_search.
API_Tag: list_of_cocktails, detailed_cocktail_recipe_by_id, newssearch.
Plan:\nThought: I'm planning a surprise birthday party for my best friend. Can you help me find a list of popular cocktails to serve at the party? Also, I need some suggestions for cocktail recipes that are easy to make. Additionally, find some interesting news articles related to birthday celebrations.

System: I'm a food enthusiast and I want to explore different cuisines. 
Category: Food, Food, Data, Finance.
Cate_Tag: Food, Food, Data, Finance.
Tool_Tag: the_cocktail_db, the_cocktail_db, web_search, investors_exchange_iex_trading.
API_Tag: list_of_cocktails, detailed_cocktail_recipe_by_id, newssearch, iex_regulation_sho_threshold_securities_list.
Plan:\nThought: I'm a food enthusiast and I want to explore different cuisines. Can you suggest some popular cocktails and their recipes that pair well with different types of cuisine? Also, find some interesting news articles about the culinary world. Additionally, provide me with the current threshold securities list for NVIDIA's stock.

System: I'm planning a family vacation to a beach destination. 
Tool: the_cocktail_db, the_cocktail_db, investors_exchange_iex_trading, live_sports_odds.
Cate_Tag: Food, Food, Finance, Sports.
Tool_Tag: the_cocktail_db, the_cocktail_db, investors_exchange_iex_trading, live_sports_odds.
API_Tag: list_of_cocktails, detailed_cocktail_recipe_by_id, ohlc, v4_sports_sport_odds.
Plan:\nThought: I'm planning a family vacation to a beach destination. Can you recommend some refreshing cocktails to enjoy by the beach? Also, provide me with the historical open and close prices for Qualcomm's stock. Furthermore, suggest some sports activities that can be enjoyed on the beach.

Now, Please make the tool using plan of below requests.
System: {request} 
Plan:'''



Whole32TagTrace_PROMPT='''Assume that you play a role of tool using planner, I would give you a user request and its corresponding tag list, and you should help me to make the tool using tag trace.
Here are some examples of human request and corresponding tool using tag trace: 

System: I'm planning a fun-filled weekend with my family and I want to start it off with a good laugh.
Cate_Tag: Data, Data, Entertainment, Entertainment.
Tool_Tag: socialgrep, socialgrep, programming_memes_reddit, reddit_meme.
API_Tag: post_search, comment_search, get_all_memes, top_memes.
Tag_Trace:\nThought: get_all_memes_for_programming_memes_reddit, post_search_for_socialgrep, comment_search_for_socialgrep, comment_search_for_socialgrep, Finish

System: I'm planning a road trip with my family and we need some good music for the journey. 
Category: Finance, Data.
Cate_Tag: Finance, Data.
Tool_Tag: global_ethereum_price_index_gex, tardis_dev.
API_Tag: short_ticker, exchanges.
Tag_Trace:\nThought: invalid_hallucination_function_name, invalid_hallucination_function_name, invalid_hallucination_function_name, invalid_hallucination_function_name, Finish

System: Please suggest a fun fact about a random year and a random NBA player's statistics. 
Tool: numbers, free_nba, chuck_norris.
Cate_Tag: Education, Sports, Social.
Tool_Tag: numbers, free_nba, chuck_norris.
API_Tag: get_random_fact, get_all_stats, jokes_search.
Tag_Trace:\nThought: get_random_fact_for_numbers, get_all_stats_for_free_nba, get_random_fact_for_numbers, jokes_search_for_chuck_norris, Finish

Now, Please make the tool using plan of below requests.
System: {request} 
Tag_Trace:'''

Mix2Whole3_PROMPT='''You are a helpful assistant and good planner. Your job is to find which APIs assistant can use by given the seed task and tools. 
First I will give you the a user request and its corresponding tools as the seed task, and your job start.

Here are some examples of human request and corresponding tools: 

System: I'm planning a surprise birthday party for my best friend.
Tool: the_cocktail_db, the_cocktail_db, web_search.
Tag: \nThought: 
Cate_Tag: Food, Food, Data.
Tool_Tag: the_cocktail_db, the_cocktail_db, web_search.
API_Tag: list_of_cocktails, detailed_cocktail_recipe_by_id, newssearch.

System: I'm a food enthusiast and I want to explore different cuisines. 
API: list_of_cocktails, detailed_cocktail_recipe_by_id, newssearch, iex_regulation_sho_threshold_securities_list.
Tag:\nThought: Cate_Tag: Food, Food, Data, Finance.
Tool_Tag: the_cocktail_db, the_cocktail_db, web_search, investors_exchange_iex_trading.
API_Tag: list_of_cocktails, detailed_cocktail_recipe_by_id, newssearch, iex_regulation_sho_threshold_securities_list.

System: I'm planning a family vacation to a beach destination. 
Category: Food, Food, Finance, Sports.
Tag:\nThought: Cate_Tag: Food, Food, Finance, Sports.
Tool_Tag: the_cocktail_db, the_cocktail_db, investors_exchange_iex_trading, live_sports_odds.
API_Tag: list_of_cocktails, detailed_cocktail_recipe_by_id, ohlc, v4_sports_sport_odds.

Now, Please make the API using plan of below requests and tools.
System: {request} 
Tag:'''




CATETOOL_PROMPT='''You are a helpful assistant and good planner. Your job is to make tool using plans to help human find tools(functions) they can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.
You need to think about whether to use tool first. If yes, make tool using plan.
Only these tool names are optional when making plans: {toolnames}
Assume that you play a role of tool using planner, I would give you a user request, and you should help me to make the tool using plan.
Here are some examples of human request and corresponding tool using plan: 

System: I'm planning a surprise birthday party for my best friend.
Category: Food, Food, Data.
Tool: The Cocktail DB, The Cocktail DB, Web Search.
Plan:\nThought: I'm planning a surprise birthday party for my best friend. Can you help me find a list of popular cocktails to serve at the party? Also, I need some suggestions for cocktail recipes that are easy to make. Additionally, find some interesting news articles related to birthday celebrations.

System: I'm a food enthusiast and I want to explore different cuisines. 
Category: Food, Food, Data, Finance.
Tool: The Cocktail DB, The Cocktail DB, Web Search, Investors Exchange (IEX) Trading.
Plan:\nThought: I'm a food enthusiast and I want to explore different cuisines. Can you suggest some popular cocktails and their recipes that pair well with different types of cuisine? Also, find some interesting news articles about the culinary world. Additionally, provide me with the current threshold securities list for NVIDIA's stock.

System: I'm planning a family vacation to a beach destination. 
Category: Food, Food, Finance, Sports.
Tool: The Cocktail DB, The Cocktail DB, Investors Exchange (IEX) Trading, Live Sports Odds.
Plan:\nThought: I'm planning a family vacation to a beach destination. Can you recommend some refreshing cocktails to enjoy by the beach? Also, provide me with the historical open and close prices for Qualcomm's stock. Furthermore, suggest some sports activities that can be enjoyed on the beach.

Now, Please make the tool using plan of below requests.
System: {request} 
Plan:'''


ToolAPI2Plan_PROMPT='''You are a helpful assistant and good planner. Your job is to make tool using full instructions plans to help human find tools(APIs) they can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.
You have access of the following tools: {toolnames}

Specifically, you have access to the following APIs: {APInames}

Here are some examples of human request and corresponding tool using plan: 

System: I'm planning a surprise birthday party for my best friend.
Tool: The Cocktail DB, The Cocktail DB, Web Search.
API: List of Cocktails, Detailed Cocktail Recipe by ID, newsSearch.
Plan:\nThought: I'm planning a surprise birthday party for my best friend. Can you help me find a list of popular cocktails to serve at the party? Also, I need some suggestions for cocktail recipes that are easy to make. Additionally, find some interesting news articles related to birthday celebrations.

System: I'm a food enthusiast and I want to explore different cuisines. 
Tool: The Cocktail DB, The Cocktail DB, Web Search, Investors Exchange (IEX) Trading.
API: List of Cocktails, Detailed Cocktail Recipe by ID, newsSearch, IEX Regulation SHO Threshold Securities List.
Plan:\nThought: I'm a food enthusiast and I want to explore different cuisines. Can you suggest some popular cocktails and their recipes that pair well with different types of cuisine? Also, find some interesting news articles about the culinary world. Additionally, provide me with the current threshold securities list for NVIDIA's stock.

System: I'm planning a family vacation to a beach destination. 
Tool: The Cocktail DB, The Cocktail DB, Investors Exchange (IEX) Trading, Live Sports Odds.
API: List of Cocktails, Detailed Cocktail Recipe by ID, OHLC, /v4/sports/{sport}/odds.
Plan:\nThought: I'm planning a family vacation to a beach destination. Can you recommend some refreshing cocktails to enjoy by the beach? Also, provide me with the historical open and close prices for Qualcomm's stock. Furthermore, suggest some sports activities that can be enjoyed on the beach.

Now, Please make the tool using plan of below requests.
System: {request} 
Plan:'''


API2Plan_PROMPT='''You are a helpful assistant and good planner. Your job is to make tool using full instructions plans to help human find tools(APIs) they can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.

Specifically, you have access to the following APIs: {APInames}

Here are some examples of human request and corresponding tool using plan: 

System: I'm planning a surprise birthday party for my best friend.
API: List of Cocktails, Detailed Cocktail Recipe by ID, newsSearch.
Plan:\nThought: I'm planning a surprise birthday party for my best friend. Can you help me find a list of popular cocktails to serve at the party? Also, I need some suggestions for cocktail recipes that are easy to make. Additionally, find some interesting news articles related to birthday celebrations.

System: I'm a food enthusiast and I want to explore different cuisines. 
API: List of Cocktails, Detailed Cocktail Recipe by ID, newsSearch, IEX Regulation SHO Threshold Securities List.
Plan:\nThought: I'm a food enthusiast and I want to explore different cuisines. Can you suggest some popular cocktails and their recipes that pair well with different types of cuisine? Also, find some interesting news articles about the culinary world. Additionally, provide me with the current threshold securities list for NVIDIA's stock.

System: I'm planning a family vacation to a beach destination. 
API: List of Cocktails, Detailed Cocktail Recipe by ID, OHLC, /v4/sports/{sport}/odds.
Plan:\nThought: I'm planning a family vacation to a beach destination. Can you recommend some refreshing cocktails to enjoy by the beach? Also, provide me with the historical open and close prices for Qualcomm's stock. Furthermore, suggest some sports activities that can be enjoyed on the beach.

Now, Please make the tool using plan of below requests.
System: {request} 
Plan:'''


Cut2Cate_PROMPT='''You are a helpful assistant and good planner. Your job is to find which category of tools assistant can use to complete the following seed task. 
First I will give you the a user request as the seed task, and your job start.
Only these categories are optional when making plans: {toolnames}

Here are some examples of human request and corresponding categories: 

System: I'm planning a surprise birthday party for my best friend.
Category:\nThought: Food, Food, Data.

System: I'm a food enthusiast and I want to explore different cuisines. 
Category:\nThought: Food, Food, Data, Finance.

System: I'm planning a family vacation to a beach destination. 
Category:\nThought: Food, Food, Finance, Sports.

Now, Please make the tool using plan of below requests.
System: {request} 
Category:'''


Cate2Tool_PROMPT='''You are a helpful assistant and good planner. Your job is to find which tools assistant can use to complete the following seed task. 
First I will give you the a user request and its corresponding categories as the seed task, and your job start.
Only these tools are optional when making plans: {toolnames}

Here are some examples of human request and corresponding categories: 

System: I'm planning a surprise birthday party for my best friend.
Category: Food, Food, Data.
Tool: \nThought: The Cocktail DB, The Cocktail DB, Web Search.

System: I'm a food enthusiast and I want to explore different cuisines. 
Category: Food, Food, Data, Finance. 
Tool:\nThought: The Cocktail DB, The Cocktail DB, Web Search, Investors Exchange (IEX).

System: I'm planning a family vacation to a beach destination. 
Category: Food, Food, Finance, Sports.
Tool:\nThought: The Cocktail DB, The Cocktail DB, Investors Exchange (IEX) Trading, Live Sports Odds.

Now, Please make the tool using plan of below requests.
System: {request} 
Tool:'''


Tool2API_PROMPT='''You are a helpful assistant and good planner. Your job is to find which APIs assistant can use by given the seed task and tools. 
First I will give you the a user request and its corresponding tools as the seed task, and your job start.
Only these APIs are optional when making plans: {toolnames}

Here are some examples of human request and corresponding tools: 

System: I'm planning a surprise birthday party for my best friend.
Tool: The Cocktail DB, The Cocktail DB, Web Search.
API: \nThought: List of Cocktails, Detailed Cocktail Recipe by ID, newsSearch.

System: I'm a food enthusiast and I want to explore different cuisines. 
Tool: The Cocktail DB, The Cocktail DB, Web Search, Investors Exchange (IEX). 
API:\nThought: List of Cocktails, Detailed Cocktail Recipe by ID, newsSearch, IEX Regulation SHO Threshold Securities List.

System: I'm planning a family vacation to a beach destination. 
Tool: The Cocktail DB, The Cocktail DB, Investors Exchange (IEX) Trading, Live Sports Odds.
API:\nThought: List of Cocktails, Detailed Cocktail Recipe by ID, OHLC, /v4/sports/{sport}/odds.

Now, Please make the API using plan of below requests and tools.
System: {request} 
API:'''