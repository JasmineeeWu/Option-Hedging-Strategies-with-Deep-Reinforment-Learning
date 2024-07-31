# Delta Hedging with and without transaction costs
delta = test_data['delta'].values
bid_price = test_data['bid'].values
ask_price = test_data['ask'].values

position = 0
transaction_cost_percentage = 0.01
total_pnl_with_costs = 0

for t in range(len(test_data)):
    current_delta = delta[t]
    current_bid_price = bid_price[t]
    current_ask_price = ask_price[t]

    if current_delta > 0.5:
        transaction_cost = transaction_cost_percentage * (current_ask_price - current_bid_price)
        pnl = current_ask_price - transaction_cost - current_bid_price
        position += 1
    elif current_delta < -0.5:
        transaction_cost = transaction_cost_percentage * (current_ask_price - current_bid_price)
        pnl = current_bid_price - transaction_cost - current_ask_price
        position -= 1
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

    if current_delta > 0.5:
        pnl = current_ask_price - current_bid_price
        position += 1
    elif current_delta < -0.5:
        pnl = current_bid_price - current_ask_price
        position -= 1
    else:
        pnl = 0

    total_pnl_without_costs += pnl

print(f"Total P&L without transaction costs: {total_pnl_without_costs}")
