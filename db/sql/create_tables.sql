-- Create prices table
create table if not exists stock_prices_data(
datetime timestamp with time zone not null,
symbol text not null,
close_price double precision not null,
volume double precision not null
);

-- Make hypertable from it
select create_hypertable('stock_prices_data', 'datetime');

-- 
select * from stock_prices_data;
