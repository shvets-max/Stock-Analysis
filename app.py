
from flask import Flask, render_template, request
import yfinance as yf
from yfinance.const import fundamentals_keys, SECTOR_INDUSTY_MAPPING
import psycopg2

connection = psycopg2.connect(
    user="postgres",
    password="password",
    host="127.0.0.1",
    port="5432",
    database="postgres"
)
cursor = connection.cursor()

app = Flask(__name__)
# Route for home page where user selects country

@app.route('/')
def select_sector():
    return render_template('select_sector.html', sectors=SECTOR_INDUSTY_MAPPING.keys())

# Route for selecting industry based on chosen country
@app.route('/select_industry', methods=['POST'])
def select_industry():
    sector = request.form['sector']
    return render_template(
        'select_industry.html',
        sector=sector,
        industries=SECTOR_INDUSTY_MAPPING[sector]
    )

# Route for selecting stock based on chosen country and industry
@app.route('/select_stock', methods=['POST'])
def select_stock():
    sector = request.form['sector']
    industry = yf.Industry(request.form['industry'])

    available_stocks = (
            industry.top_performing_companies.index.tolist()
            + industry.top_growth_companies.index.tolist()
    )
    return render_template(
        'select_stock.html',
        sector=sector,
        industry=industry.name,
        stocks=available_stocks
    )

# Route for setting time range for selected stock
@app.route('/set_time_range', methods=['POST'])
def set_time_range():
    stock = request.form['stock']
    return render_template('set_time_range.html', stock=stock)

# Route for displaying stock prices
@app.route('/show_stock_data', methods=['POST'])
def show_stock_data():
    stock = request.form['stock']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    stock_data = fetch_stock_data(stock, start_date, end_date)

    return render_template('show_stock_data.html', stock=stock, start_date=start_date, end_date=end_date, stock_data=stock_data)

def fetch_stock_data(stock, start_date, end_date):
    ticker = yf.Ticker(stock)
    hist = ticker.history(start=start_date, end=end_date)

    query = "INSERT INTO stock_prices_data (datetime, symbol, close_price, volume)" +\
        " VALUES (%s,%s,%s,%s)"

    for _, raw in hist.iterrows():
        record_to_insert = (raw.name.to_pydatetime(), stock, float(raw["Close"]), float(raw["Volume"]))
        cursor.execute(query, record_to_insert)
        connection.commit()


    return hist.to_dict(orient="index")


if __name__ == '__main__':
    app.run(debug=True)
