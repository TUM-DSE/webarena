
categories = {}

categories['Data Analysis'] = [
        'What is the top-{{n}} best-selling product in {{year}}',
        'What is the top-{{n}} best-selling brand in {{period}}',
        'What is the top-{{n}} best-selling product type in {{period}}',
        'What is the price range of {{product}} in the One Stop Market?',
        'What is the price range for products from {{brand}}?',
        'I am doing a market survey for one stop market, show me the most expensive product from {{product_category}} category',
        'What is the rating of {{product}}',
        'Summarize customer reviews for {{product}}',
        'What are the main criticisms of this product? Please extract the relevant sentences',
        'Tell me the reasons why customers like {{product}}',
        'Show me the customers who have expressed dissatisfaction with {{product}}',
        'What are the key aspects that the customers don\'t like about {{product}}',
        'What do customers say about {{product_type}} from {{manufature}}',
        'Who gave {{stars}} for phone cases from EYZUTAK',
        'List the customer names who complain about the quality of EYZUTAK phone cases',
        'List the customer names who thinks EYZUTAK phone cases are of good looking',
        'Tell me the count of comments that have received more downvotes than upvotes',
        'What is the total count of {{status}} reviews amongst all the reviews?',
        'How many reviews our shop received {{time}}?',
        'List out reviewers, if exist, who mention about {{description}}',
        'Tell me the the number of reviews that our store received by far that mention term "{{term}}"'
        'Get the total payment amount of the last {{N}} {{status}} orders',
        'Compare the payment difference of the last {{N}} {{status_1}} orders and {{status_2}} orders',
        'What\'s the total number of items sold in the most recent {{k}} orders?',
        'Presents the monthly count of successful orders {{period}} in MM:COUNT format',
        'Which customer has completed the {{quantifier}} number of orders in the entire history?',
        'Which customer has placed {{number}} orders in the entire history?',
        'Tell me the {{attribute}} of the customer who has the most cancellations in the history',
        'Show me the {{information}} of the customer who is the most unhappy with {{product}}',
        'List the top {{n}} search terms in my store',
        'What brands appear most frequently among the top search terms?',
        'Give me the {{Attribute}} of the products that have {{N}} units left',
        'Gather the titles of {{product}} reviews with {{rating}} rating from OneStopShop',
        'Today is 3/15/2023, generate a {{report}} {{time_span}}',
        'Create an {{type}} report from {{start_date}} to {{end_date}}',
        'Among the top {{number}} post in "{{subreddit}}" forum, {{description}}',
        'Tell me the count of comments that have received more downvotes than upvotes for the user who made the latest post on the {{forum}} forum',
        'Gather the titles of {{product}} reviews with {{rating}} rating from OneStopShop, and post them in the games subreddit under the title "real user feedback on {{product}}"',
        'What are the top-{{n}} best-selling product in {{year}}',
        'What are the top-{{n}} best-selling product in {{period}}',
        'What are the top-{{n}} best-selling product in {{year}}',
        'Tell me the the number of reviews that our store received by far that mention term "{{term}}"'
        'Tell me the count of comments that have received more downvotes than upvotes for the user who made the latest post on the {{forum}} forum.',
        'Tell me the the number of reviews that our store received by far that mention term "{{term}}"',
        'Tell me the count of comments that have received more downvotes than upvotes for the user who made the latest post on the {{forum}} forum.',




        ]

categories['Account Management'] = [
        'What is the date when I made my first purchase on this site?',
        'Today is 6/12/2023. Tell me how many fulfilled orders I have {{period}}, and the total amount of money I spent',
        'Today is 6/12/2023. Tell me how many fulfilled orders I have {{period}}, and the total amount of money I spent.',
        'I recently moved, my address is {{address}}, update my information on OneStopShopping accordingly',
        'Update the project site\'s title to "{{title}}"',
        'Change the page title of "{{old-heading}}" page on my site to "{{heading}}"',
        'Mark all {{brand}} shirts on sale',
        'Disable {{product}} from the site, they are facing some quality issues',
        '{{action}} the price of this product by {{amount}}',
        '{{action}} the price of {{config}} by {{amount}}',
        'Make all {{product}} as out of stock',
        '{{quantity}} {{product}} arrived, update the stock',
        'We\'ve received {{quantity}} {{product}}, please update the inventory',
        'We\'ve received {{quantity}}, update the inventory',
        'Add a simple product named {{product}} with {{stock}} in stock, available in size {{size}} and color {{color}}, priced at ${{price}}',
        'Add a new {{option}} option {{value}} to the {{base_setting}} of {{product}}',
        'Update the product description of {{product}} to highlight the real user positive reviews by quoting the comments',
        'Update the description of {{product}} to highlight the real user positive reviews by quoting the comments',
        'Change my reddit bio to "{{content}}"',


        ]

categories['Order Management'] = [
        'Get the {{attribute}} of the {{status}} order',
        'Get the order number of my most recent {{status}} order',
        'Tell me the status of my latest order and when will it arrive',
        'Show the most recent {{status}} order',
        'Tell me the total cost of my latest {{status}} order?',
        'Tell me when I last ordered my {{description}}?',
        'Show me the {{info}} for order number {{order_number}}',
        'Cancel order {{id}}',
        'Change the delivery address for my most recent order to {{address}}',
        'Modify the address of order #{{order_id}} to {{address}}',
        'Notify {{name}} in their most recent pending order with message "{{message}}"',
        'Update order #{{order}} with the {{service}} tracking number {{tracking}}',


        ]

categories['Social Media'] = [
        'Rate my recent purchase of {{product}} with {{num_star}} stars, using my nickname {{nickname}}?',
        'Post a review of my recent reading "{{book}}" in the r/books with my comment "{{content}}"',
        'Re-post the image of {{content}} in this page to {{subreddit}} subreddit and note "from /f/pics"',
        'Create a discussion post about "{{topic}}" in a relevant subreddit and ask users for their opinions',
        'Post a notice on a virtual meetup for {{interest}} enthusiasts on {{date}} in the {{subreddit}} subreddit',
        'Post in {{subreddit}} subreddit about what could machine learning help the correpong field',
        'Post in {{subreddit}} subreddit about what could midjourney help the correpong field',
        'Post in {{subreddit}} forum about what could open-source LLMs help the correpong field',
        'Post in {{subreddit}} forum about what could large language models help the correpong field',
        'Post in {{subreddit}} subreddit about what could diffusion model help the correpong field',
        'Edit my post on {{post}} by adding a line to the body that says "{{content}}"',
        'Post my question, "{{question}}", in a subreddit where I\'m likely to get an answer',
        'Find a subreddit focused on topics related to {{topic}}, and post my question, "{{question}}" there',
        'Ask for advice about {{issue}} in a subreddit for relations',
        'Ask for product recommendations for {{category}} within a budget of {{price}} in {{subreddit}}',
        'Post in the most appropriate subreddit and ask for recommendations for {{category}} products within a budget of {{price}}',
        'Upvote the newest post in {{subreddit}} subreddit',
        'Reply to {{position_description}} in this post with "{{content_description}}"',
        'Reply to {{position_description}} with my comment "{{content_description}}"',
        'Thumbs down the top {{k}} post ever in {{subreddit}}',
        'Like all submissions created by {{user}} in subreddit {{subreddit}}',
        'DisLike all submissions created by {{user}} in subreddit {{subreddit}}',
        'Open the thread of a trending post on the forum "{{subreddit}}" and subscribe',
        'Create a new forum named {{name}}, with a description of {{description}}, and include {{sidebar_list}} in the sidebar?',

        ]

categories['Personal Assistant'] = [
        'I have jaw bruxism problem, show me something that could alleviate the problem',
        'What is the {{option}} configuration of the {{product}} I bought {{time}}',
        'How much I spent on {{category}} shopping during {{time}}',
        'How much I spend {{time}} on shopping at One Stop Market?',
        'How much did I spend on shopping at One Stop Market {{time}}?',
        'Lookup orders that are {{status}}',
        'Telll me the grand total of invoice {{id}}',
        'How much refund I should expect from my order canlled in {{time}}, including shipping fee',
        'How much refund I should expect from my order canlled in {{time}} if I cannot get the shipping fee refunded?',
        'Draft a refund message via their "contact us" form for the {{product}} I bought {{time}}',
        'Fill the "contact us" form in the site for a refund on the {{product}} I bought',
        'Find the customer name and email with phone number {{PhoneNum}}',
        'Show all customers',
        'Which number to call for the customer service?',
        'Subscribe to the newsletter of OneStopMarket',
        'Preview the {{name}} theme for my shop',
        'Draft a new marketing price rule for {{topic}} that offers {{rule}} for all customers',
        'Draft an email to the shop owner via their contact us function for a coupon as {{reason}}',
        'Approve the positive reviews to display in our store',
        'Delete all {{review_type}}',
        'Check out my todos',

        ]

categories['Shopping Assistant'] = [
        'Search for "{{keyword}}"',
        'I want to browse the products in the {{category}} category',
        'Show me products under ${{price}} in "{{product_category}}" category',
        'List products from {{product_category}} category by {{order}} price',
        'Find discounted items',
        'Show me the "{{product}}" listings by {{sorting_order}}',
        'Look up the most recent models of XBox controllers released between 2020-2021?',
        'Show the least expensive {{product}} with a minimum storage capacity of {{min_storage}}',
        'I have a lot of Nintendo Switch game cards now, help me find the best storage option to fit all {{num}} cards',
        'Add the product with the lowest per unit price from my open tabs to the shopping cart',
        'Add {{product}} to my wish list',
        'Add a {{product}} to my wish list',
        'Add this product to my wishlist',
        'Buy the highest rated product from the {{product_category}} category within a budget {{dollar_value}}',
        'Buy the best rating product from "{{category}}" category with at least 5 reviews and the product is least expensive',
        'I previously ordered some {{product}} {{time}} and later cancelled. Can you reorder it for me?',
        'Provide me with the complete names of Bluetooth headphones from Sony, and also share the price range for the available models',
        'Provide me with the full names of chargers from Anker, and also share the price range for the available models',
        'Please provide me with the complete product names of Oral B brush heads designed for children, along with their corresponding price range per brush',
        'List the full product names of slide slippers from Nike and tell me the price range of the available products',



        ]
