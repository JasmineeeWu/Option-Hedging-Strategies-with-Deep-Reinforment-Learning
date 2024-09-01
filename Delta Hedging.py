delta = test_data['delta'].values
last_price = test_data['last'].values

position = 0
cumulative_pnl_with_costs = []
transaction_cost_percentage = 0.5

total_pnl_with_costs = 0
for t in range(1, len(test_data)):
    current_delta = delta[t]
    previous_last_price = last_price[t - 1]
    current_last_price = last_price[t]

    transaction_cost = transaction_cost_percentage * abs(current_last_price - previous_last_price)

    if current_delta > position:
        change_in_position = current_delta - position
        pnl = (current_last_price - previous_last_price) - transaction_cost * change_in_position
        position = current_delta 
    elif current_delta < position:  
        change_in_position = position - current_delta
        pnl = (previous_last_price - current_last_price) - transaction_cost * change_in_position
        position = current_delta 
    else:
        pnl = 0 

    total_pnl_with_costs += pnl
    cumulative_pnl_with_costs.append(total_pnl_with_costs)

position = 0
cumulative_pnl_without_costs = []

total_pnl_without_costs = 0
for t in range(1, len(test_data)):
    current_delta = delta[t]
    previous_last_price = last_price[t - 1]
    current_last_price = last_price[t]

    if current_delta > position: 
        pnl = current_last_price - previous_last_price
        position = current_delta 
    elif current_delta < position: 
        pnl = previous_last_price - current_last_price
        position = current_delta  
    else:
        pnl = 0 
    total_pnl_without_costs += pnl
    cumulative_pnl_without_costs.append(total_pnl_without_costs)


print(f"Total P&L with transaction costs: {total_pnl_with_costs:.4f}")
print(f"Total P&L without transaction costs: {total_pnl_without_costs:.4f}")
