% Manually copy the results folder from the run into your MATLAB directory.

% Get a list of the files I'm interested in
cd('results');
listings = dir('account_value_trade_ensemble_*.csv');

% Sort the csv's by the time they were created (most recent first)

% Convert struct to table
T = struct2table(listings);

% Sort table
T_sorted = sortrows(T, 'date');

% Convert table back to struct
list_sorted = table2struct(T_sorted);

% Read and concatenate portfolio values
daily_value = zeros(63*length(list_sorted),1);

for i = 1:length(list_sorted)
    % Load csv
    csv_data = readmatrix(list_sorted(i).name);
    
    % Put prices into daily_value
    daily_value(((i-1)*63 + 1):(i*63)) = csv_data(2:end,2);
    
end

% Export daily portfolio values to csv
csvwrite('daily_portfolio_value.csv', daily_value);

% Plot for me
indices = 1:length(daily_value);

plot(indices,daily_value)