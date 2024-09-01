delta = test_data['delta'].values
bid_price = test_data['bid'].values
ask_price = test_data['ask'].values
last_price = test_data['last'].values

position = 0
delta_transaction_cost = 0.5
total_pnl_with_costs = 0

for t in range(len(test_data)):
    current_delta = delta[t]
    current_bid_price = bid_price[t]
    current_ask_price = ask_price[t]
    current_last_price = last_price[t]
    previous_last_price = last_price[t - 1]

    
    transaction_cost = delta_transaction_cost * (current_ask_price - current_bid_price)

    if current_delta > 0.5:
        pnl = current_last_price - previous_last_price - transaction_cost
        position += 1
    elif current_delta < -0.5:
        pnl = previous_last_price - current_last_price - transaction_cost
        position -= 1
    else:
        if position > 0:  
            pnl = current_last_price - last_price[t - 1] if t > 0 else 0
        elif position < 0:  
            pnl = last_price[t - 1] - current_last_price if t > 0 else 0
        else:
            pnl = 0

    total_pnl_with_costs += pnl

print(f"Total P&L with transaction costs: {total_pnl_with_costs}")

position = 0
total_pnl_without_costs = 0

for t in range(len(test_data)):
    current_delta = delta[t]
    current_bid_price = bid_price[t]
    current_ask_price = ask_price[t]
    current_last_price = last_price[t]
    previous_last_price = last_price[t - 1]

    if current_delta > 0.5:
        pnl = current_last_price - previous_last_price
        position += 1
    elif current_delta < -0.5:
        pnl = previous_last_price - current_last_price
        position -= 1
    else:
        if position > 0: 
            pnl = current_last_price - last_price[t - 1] if t > 0 else 0
        elif position < 0: 
            pnl = last_price[t - 1] - current_last_price if t > 0 else 0
        else:
            pnl = 0

    total_pnl_without_costs += pnl

print(f"Total P&L without transaction costs: {total_pnl_without_costs}")
