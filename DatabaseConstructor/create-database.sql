CREATE DATABASE stonksdb;

\c stonksdb

CREATE TABLE metadata_currency(
    id SERIAL PRIMARY KEY,
    currency VARCHAR NOT NULL,
    CONSTRAINT currency_unique UNIQUE (currency)
);

INSERT INTO 
    metadata_currency(currency)
VALUES
    ('DKK');

CREATE TABLE metadata_asset(
    id SERIAL PRIMARY KEY,
    asset VARCHAR(100) NOT NULL,
    CONSTRAINT asset_unique UNIQUE(asset)
);

INSERT INTO
    metadata_asset(asset)
VALUES
    ('Stock');

CREATE TABLE metadata_exchange(
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(100) NOT NULL,
    CONSTRAINT exchange_unique UNIQUE(exchange)
);

INSERT INTO 
    metadata_exchange(exchange)
VALUES
    ('CSE');

CREATE TABLE metadata_summary(
    id SERIAL PRIMARY KEY,
    summary VARCHAR(100) NOT NULL,
    CONSTRAINT summary_unique UNIQUE(summary)
);

INSERT INTO 
    metadata_summary(summary)
VALUES
    ('Instrument');

CREATE TABLE metadata_issuer(
    id SERIAL PRIMARY KEY,
    country VARCHAR(100),
    CONSTRAINT country_unique UNIQUE(country)
);

INSERT INTO
    metadata_issuer(country)
VALUES
    ('DK');

CREATE TABLE metadata_category(
    id SERIAL PRIMARY KEY,
    category VARCHAR(100),
    CONSTRAINT category_unique UNIQUE(category)
);

INSERT INTO
    metadata_category(category)
VALUES
    ('UNKNOWN');

CREATE TABLE dataset(
    Identifier INT PRIMARY KEY,
    AssetType VARCHAR(100) NOT NULL,
    CurrencyCode VARCHAR(100),
    Description VARCHAR(100) NOT NULL,
    ExchangeId VARCHAR(100) NOT NULL,
    GroupId VARCHAR(100) NOT NULL,
    IssuerCountry VARCHAR(100) NOT NULL,
    PrimaryListing VARCHAR(100) NOT NULL,
    SummaryType VARCHAR(100) NOT NULL,
    Symbol VARCHAR(100) NOT NULL,
    Category VARCHAR(100) NOT NULL,

    CONSTRAINT fk_asset
        FOREIGN KEY(AssetType)
            REFERENCES metadata_asset(asset),

    CONSTRAINT fk_currency
        FOREIGN KEY(CurrencyCode)
            REFERENCES metadata_currency(currency),
    
    CONSTRAINT fk_exchange
        FOREIGN KEY(ExchangeId)
            REFERENCES metadata_exchange(exchange),
    
    CONSTRAINT fk_issuer
        FOREIGN KEY(IssuerCountry)
            REFERENCES metadata_issuer(country),
    
    CONSTRAINT fk_category
        FOREIGN KEY(Category)
            REFERENCES metadata_category(category)
);

CREATE TABLE stock(
    id SERIAL PRIMARY KEY,
    Identifier INT NOT NULL,
    Close DECIMAL NOT NULL,
    High DECIMAL NOT NULL,
    Interest DECIMAL NOT NULL,
    Low DECIMAL NOT NULL,
    Open DECIMAL NOT NULL,
    Time DATE NOT NULL,
    Volume DECIMAL NOT NULL
);

CREATE INDEX stock_id_index ON stock(Identifier, Time);

CREATE TYPE stock_type AS (
    Identifier INT,
    Close DECIMAL,
    High DECIMAL,
    Interest DECIMAL,
    Low DECIMAL,
    Open DECIMAL,
    Time DATE,
    Volume DECIMAL
);

CREATE FUNCTION upsert_stock
(
    source stock_type
)
RETURNS VOID
as $$
BEGIN
    UPDATE stock SET Close = source.Close, High = source.High, Interest = source.Interest, Low = source.Low, Open = source.Open, Volume = source.Volume WHERE Identifier = source.Identifier AND Time = source.Time;
    IF NOT FOUND THEN
    INSERT INTO stock(Identifier, Close, High, Interest, Low, Open, Time, Volume)
    VALUES(source.Identifier, source.Close, source.High, source.Interest, source.Low, source.Open, source.Time, source.Volume);
    END IF;
end; $$
language plpgsql
