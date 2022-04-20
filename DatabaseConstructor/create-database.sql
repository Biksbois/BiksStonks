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
            PrimaryCategory VARCHAR(100),
            SecondaryCategory VARCHAR(100)
        ); 
    END IF;
END
$$;

CREATE TABLE IF NOT EXISTS sentiment_dataset(
    id SERIAL PRIMARY KEY,
    translator VARCHAR(100),
    source_language VARCHAR(10),
    target_language VARCHAR(10),
    source VARCHAR(20),
    url VARCHAR(40),
    description VARCHAR(200),
    category VARCHAR(30)
);

CREATE TABLE IF NOT EXISTS sentiment(
    id SERIAL PRIMARY KEY,
    datasetid INT,
    release_date TIMESTAMP,
    source_headline VARCHAR(2000),
    target_headline VARCHAR(2001),
    neg DECIMAL,
    pos DECIMAL,
    neu DECIMAL,
    compound DECIMAL,
    url VARCHAR(2002),
    companies VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS model(
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    description VARCHAR(300),
    CONSTRAINT name_unique UNIQUE (name)
);

CREATE OR REPLACE FUNCTION upsert_model
(
    in_model_name VARCHAR(100),
    in_description VARCHAR(300)
)
RETURNS INT
as $$
DECLARE
    datasetid INT;
BEGIN
    UPDATE model SET description = in_description WHERE name = in_model_name;
    IF NOT FOUND THEN
        INSERT INTO model(name, description) VALUES (in_model_name, in_description);
    END IF;
    SELECT id INTO datasetid FROM model WHERE name = in_model_name;
    RETURN datasetid;
end; $$
language plpgsql;

CREATE TABLE IF NOT EXISTS score(
    id SERIAL PRIMARY KEY,
    model_id INT,
    time TIMESTAMP DEFAULT Now(),
    mae DECIMAL,
    mse DECIMAL,
    r_squared DECIMAL,
    data_from TIMESTAMP,
    data_to TIMESTAMP,
    time_unit VARCHAR(10),
    forecasted_company VARCHAR(100),
    metadata JSON NOT NULL,
    use_sentiment BOOLEAN,
    used_companies VARCHAR(100)[],
    columns VARCHAR(100)[]
);

CREATE OR REPLACE FUNCTION upsert_score
(
    in_model_id INT,
    in_time TIMESTAMP,
    in_mae DECIMAL,
    in_mse DECIMAL,
    in_r_squared DECIMAL,
    in_data_from TIMESTAMP,
    in_data_to TIMESTAMP,
    in_time_unit VARCHAR(10),
    in_forecasted_company VARCHAR(100),
    in_metadata json,
    in_use_sentiment BOOLEAN,
    in_used_companies VARCHAR(100)[],
    in_columns VARCHAR(100)[]
)
RETURNS INT
as $$
DECLARE
    datasetid INT;
BEGIN
    INSERT INTO score (model_id, time, mae, mse, r_squared, data_from, data_to, time_unit, forecasted_company, metadata, use_sentiment, used_companies, columns) VALUES (in_model_id, in_time, in_mae, in_mse, in_r_squared, in_data_from, in_data_to, in_time_unit, in_forecasted_company, in_metadata, in_use_sentiment, in_used_companies, in_columns);
    SELECT id INTO datasetid FROM score WHERE model_id = in_model_id AND time = in_time;
    RETURN datasetid;
end; $$
language plpgsql;

CREATE TABLE IF NOT EXISTS graph(
    id SERIAL PRIMARY KEY,
    score_id INT,
    y DECIMAL,
    y_hat DECIMAL,
    time TIMESTAMP
);

CREATE OR REPLACE FUNCTION upsert_graph
(
    in_score_id INT,
    in_y DECIMAL,
    in_y_hat DECIMAL,
    in_time TIMESTAMP
)
RETURNS VOID
as $$
BEGIN
    INSERT INTO graph (score_id, y, y_hat, time) VALUES (in_score_id, in_y, in_y_hat, in_time);
end; $$
language plpgsql;

CREATE TABLE IF NOT EXISTS token(
    id SERIAL PRIMARY KEY,
    value VARCHAR(600),
    valid_from TIMESTAMP,
    valid_to TIMESTAMP
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
    PrimaryCategory VARCHAR(100) NOT NULL,
    SecondaryCategory VARCHAR(100) NOT NULL,

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
    
    CONSTRAINT fk_primary_category
        FOREIGN KEY(SecondaryCategory)
            REFERENCES metadata_category(category),

    CONSTRAINT fk_secondary_category
        FOREIGN KEY(SecondaryCategory)
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
CREATE INDEX IF NOT EXISTS sentiment_index ON sentiment(datasetid, companies, release_date);
CREATE INDEX IF NOT EXISTS sentiment_headline_index ON sentiment(source_headline);



CREATE OR REPLACE FUNCTION upsert_sentiment_dataset
(
    in_translator VARCHAR(100),
    in_source_language VARCHAR(10),
    in_target_language VARCHAR(10),
    in_source VARCHAR(20),
    in_url VARCHAR(40),
    in_description VARCHAR(200),
    in_category VARCHAR(30)

)
RETURNS INT
as $$
DECLARE
    datasetid INT;
BEGIN
    UPDATE sentiment_dataset SET url = in_url, description = in_description  WHERE translator = in_translator AND source_language = in_source_language AND target_language = in_target_language AND source = in_source AND category = in_category;
    IF NOT FOUND THEN
    INSERT INTO sentiment_dataset(translator, source_language, target_language, source, url, description, category)
    VALUES (in_translator, in_source_language, in_target_language, in_source, in_url, in_description, in_category);
    END IF; 
    SELECT id INTO datasetid FROM sentiment_dataset WHERE translator = in_translator AND source_language = in_source_language AND target_language = in_target_language AND source = in_source AND category = in_category LIMIT 1;
    RETURN datasetid;
end; $$
language plpgsql;

CREATE OR REPLACE FUNCTION upsert_sentiment
(
    in_datasetid INT,
    in_release_date TIMESTAMP,
    in_source_headline VARCHAR(500),
    in_target_headline VARCHAR(500),
    in_neg DECIMAL,
    in_pos DECIMAL,
    in_neu DECIMAL,
    in_compound DECIMAL,
    in_url VARCHAR(500),
    in_companies VARCHAR(100)
)
RETURNS VOID
as $$
BEGIN
    UPDATE sentiment SET release_date = in_release_date, target_headline = in_target_headline, neg = in_neg, pos = in_pos, neu = in_neu, compound = in_compound, url = in_url, companies = in_companies WHERE source_headline = in_source_headline AND datasetid = in_datasetid;
    IF NOT FOUND THEN
    INSERT INTO sentiment(datasetid, release_date, source_headline, target_headline, neg, pos, neu, compound, url, companies) 
    VALUES (in_datasetid, in_release_date, in_source_headline, in_target_headline, in_neg, in_pos, in_neu, in_compound, in_url, in_companies);
    END IF;
end; $$
language plpgsql;

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
        UPDATE dataset SET AssetType = s.AssetType, CurrencyCode = s.CurrencyCode, Description = s.Description, ExchangeId = s.ExchangeId, GroupId = s.GroupId, IssuerCountry = s.IssuerCountry, PrimaryListing = s.PrimaryListing, SummaryType = s.SummaryType, Symbol = s.Symbol, PrimaryCategory = s.PrimaryCategory, SecondaryCategory = s.SecondaryCategory WHERE Identifier = s.Identifier;
        IF NOT FOUND THEN
        INSERT INTO dataset(Identifier, AssetType, CurrencyCode, Description, ExchangeId, GroupId, IssuerCountry, PrimaryListing, SummaryType, Symbol, PrimaryCategory, SecondaryCategory)
        VALUES(s.Identifier, s.AssetType, s.CurrencyCode, s.Description, s.ExchangeId, s.GroupId, s.IssuerCountry, s.PrimaryListing, s.SummaryType, s.Symbol, s.PrimaryCategory, s.SecondaryCategory);
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
    ('FI') ON CONFLICT (country) DO NOTHING;

INSERT INTO 
    metadata_currency(currency)
VALUES
    ('DKK') ON CONFLICT (currency) DO NOTHING;


INSERT INTO
    metadata_asset(asset)
VALUES
    ('Stock') ON CONFLICT (asset) DO NOTHING;

INSERT INTO 
    metadata_exchange(exchange)
VALUES
    ('CSE') ON CONFLICT (exchange) DO NOTHING;


INSERT INTO 
    metadata_summary(summary)
VALUES
    ('Instrument') ON CONFLICT (summary) DO NOTHING;

INSERT INTO
    metadata_category(category)
VALUES
    ('industry'),
    ('factory'),
    ('hardware_store'),
    ('construction'),
    ('commodities'),
    ('energy'),
    ('medicin'),
    ('biotech'),
    ('equipment'),
    ('drugs'),
    ('entertainment'),
    ('sport'),
    ('parks'),
    ('finance'),
    ('investing'),
    ('real_estate'),
    ('bank'),
    ('logistic'),
    ('ensurance'),
    ('it'),
    ('consultant'),
    ('detail'),
    ('tech'),
    ('end-to-end'),
    ('store'),
    ('brewery'),
    ('media_group'),
    ('UNKNOWN') ON CONFLICT (category) DO NOTHING;
