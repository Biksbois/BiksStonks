SELECT 'CREATE DATABASE stonksdb' WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'stonksdb')\gexec

\c stonksdb

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'model_type') THEN
        CREATE TYPE model_type AS(
            name VARCHAR(100),
            description VARCHAR(150)
        );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'metadata_metric_type') THEN
        CREATE TYPE metadata_metric_type AS (
            name VARCHAR(100)
        );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'score_type') THEN
        CREATE TYPE score_type AS(
            metric VARCHAR(100),
            model VARCHAR(100),
            value DECIMAL
        );
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'stock_type') THEN
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
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'dataset_type') THEN
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
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS model(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description VARCHAR(150),
    
    CONSTRAINT name_unique UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS metadata_metric(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    CONSTRAINT metadata_name_unique UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS score(
    id SERIAL PRIMARY KEY,
    metric VARCHAR(100),
    model VARCHAR(100),
    value DECIMAL,
    time TIMESTAMP DEFAULT Now(),

    CONSTRAINT fk_metric
        FOREIGN KEY (metric)
            REFERENCES metadata_metric(name),
    
    CONSTRAINT fk_model
        FOREIGN KEY (model)
            REFERENCES model(name)
);

CREATE TABLE IF NOT EXISTS metadata_currency(
    id SERIAL PRIMARY KEY,
    currency VARCHAR(100) NOT NULL,
    CONSTRAINT currency_unique UNIQUE (currency)
);

CREATE TABLE IF NOT EXISTS metadata_asset(
    id SERIAL PRIMARY KEY,
    asset VARCHAR(100) NOT NULL,
    CONSTRAINT asset_unique UNIQUE(asset)
);

CREATE TABLE IF NOT EXISTS metadata_exchange(
    id SERIAL PRIMARY KEY,
    exchange VARCHAR(100) NOT NULL,
    CONSTRAINT exchange_unique UNIQUE(exchange)
);

CREATE TABLE IF NOT EXISTS metadata_summary(
    id SERIAL PRIMARY KEY,
    summary VARCHAR(100) NOT NULL,
    CONSTRAINT summary_unique UNIQUE(summary)
);

CREATE TABLE IF NOT EXISTS metadata_issuer(
    id SERIAL PRIMARY KEY,
    country VARCHAR(100),
    CONSTRAINT country_unique UNIQUE(country)
);

CREATE TABLE IF NOT EXISTS metadata_category(
    id SERIAL PRIMARY KEY,
    category VARCHAR(100),
    CONSTRAINT category_unique UNIQUE(category)
);

CREATE TABLE IF NOT EXISTS dataset(
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

CREATE TABLE IF NOT EXISTS stock(
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

CREATE INDEX IF NOT EXISTS stock_id_index ON stock(Identifier, Time);

CREATE OR REPLACE FUNCTION upsert_stock
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

CREATE OR REPLACE FUNCTION upsert_dataset
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

CREATE OR REPLACE FUNCTION upsert_score
(
    source score_type[]
)
RETURNS VOID
as $$
DECLARE
    s score_type;
BEGIN
    FOREACH s IN ARRAY source
    LOOP
        INSERT INTO score(metric, model, value)
        VALUES(s.name, s.model, s.value);
    END LOOP;
end; $$
language plpgsql;



CREATE OR REPLACE FUNCTION upsert_model
(
    source model_type[]
)
RETURNS VOID
as $$
DECLARE
    s model_type;
BEGIN
    FOREACH s IN ARRAY source
    LOOP
        UPDATE model SET description = s.description WHERE name = s.name;
        IF NOT FOUND THEN
            INSERT INTO model(name, description)
            VALUES(s.name, s.description);
        END IF;
    END LOOP;
end; $$
language plpgsql;

DO $$
BEGIN
    IF 0 = (SELECT COUNT(*) FROM metadata_issuer) THEN
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
    END IF;
END
$$;

DO $$
BEGIN
    IF 0 = (SELECT COUNT(*) FROM metadata_currency) THEN
        INSERT INTO 
            metadata_currency(currency)
        VALUES
            ('DKK');
    END IF;
END
$$;

DO $$
BEGIN
    IF 0 = (SELECT COUNT(*) FROM metadata_asset) THEN
        INSERT INTO
            metadata_asset(asset)
        VALUES
            ('Stock');
    END IF;
END
$$;

DO $$
BEGIN
    IF 0 = (SELECT COUNT(*) FROM metadata_exchange) THEN
        INSERT INTO 
            metadata_exchange(exchange)
        VALUES
            ('CSE');
    END IF;
END
$$;

DO $$
BEGIN
    IF 0 = (SELECT COUNT(*) FROM metadata_summary) THEN
        INSERT INTO 
            metadata_summary(summary)
        VALUES
            ('Instrument');
    END IF;
END
$$;

DO $$
BEGIN
    IF 0 = (SELECT COUNT(*) FROM metadata_category) THEN
        INSERT INTO
            metadata_category(category)
        VALUES
            ('UNKNOWN');
    END IF;
END
$$;