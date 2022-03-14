CREATE DATABASE stonksdb;

\c stonksdb

CREATE TABLE IF NOT EXISTS model(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description VARCHAR(150),
    
    CONSTRAINT name_unique UNIQUE (name)
)

CREATE TABLE metadata_metric(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    CONSTRAINT name_unique UNIQUE (name)
)

CREATE TABLE metric(
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    model_name VARCHAR(100),
    value DECIMAL,
    time TIMESTAMP,

    CONSTRAINT fk_metric_name
        FOREIGN KEY (metric_name)
            REFERENCES metadata_metric(name)
    
    CONSTRAINT fk_model_name
        FOREIGN KEY (model_name)
            REFERENCES model(name)
)

CREATE TABLE metadata_currency(
    id SERIAL PRIMARY KEY,
    currency VARCHAR(100) NOT NULL,
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
    ('DK'),
    ('SE'),
    ('GB'),
    ('CH'),
    ('IS'),
    ('FO'),
    ('BS'),
    ('FR'),
    ('GL'),
    ('FI');

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
    Time TIMESTAMP NOT NULL,
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
    Time TIMESTAMP,
    Volume DECIMAL
);

CREATE TYPE dataset_type AS (
    AssetType VARCHAR(100),
    CurrencyCode VARCHAR(100),
    Description VARCHAR(100),
    ExchangeId VARCHAR(100),
    GroupId VARCHAR(100),
    Identifier INT,
    IssuerCountry VARCHAR(100),
    PrimaryListing VARCHAR(100),
    SummaryType VARCHAR(100),
    Symbol VARCHAR(100),
    Category VARCHAR(100)
);

CREATE FUNCTION upsert_stock
(
    source stock_type[]
)
RETURNS VOID
as $$
DECLARE
    s stock_type;
BEGIN
    FOREACH s IN ARRAY source
    LOOP
        UPDATE stock SET Close = s.Close, High = s.High, Interest = s.Interest, Low = s.Low, Open = s.Open, Volume = s.Volume WHERE Identifier = s.Identifier AND Time = s.Time;
        IF NOT FOUND THEN
        INSERT INTO stock(Identifier, Close, High, Interest, Low, Open, Time, Volume)
        VALUES(s.Identifier, s.Close, s.High, s.Interest, s.Low, s.Open, s.Time, s.Volume);
        END IF;
    END LOOP;
end; $$
language plpgsql;

CREATE FUNCTION upsert_dataset
(
    source dataset_type[]
)
RETURNS VOID
as $$
DECLARE
    s dataset_type;
BEGIN
    FOREACH s IN ARRAY source
    LOOP
        UPDATE dataset SET AssetType = s.AssetType, CurrencyCode = s.CurrencyCode, Description = s.Description, ExchangeId = s.ExchangeId, GroupId = s.GroupId, IssuerCountry = s.IssuerCountry, PrimaryListing = s.PrimaryListing, SummaryType = s.SummaryType, Symbol = s.Symbol, category = s.Category WHERE Identifier = s.Identifier;
        IF NOT FOUND THEN
        INSERT INTO dataset(Identifier, AssetType, CurrencyCode, Description, ExchangeId, GroupId, IssuerCountry, PrimaryListing, SummaryType, Symbol, Category)
        VALUES(s.Identifier, s.AssetType, s.CurrencyCode, s.Description, s.ExchangeId, s.GroupId, s.IssuerCountry, s.PrimaryListing, s.SummaryType, s.Symbol, s.Category);
        END IF;
    END LOOP;
end; $$
language plpgsql;
