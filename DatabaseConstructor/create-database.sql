CREATE DATABASE stonksdb

\c stonksdb

CREATE TABLE meta.currency(
    id SERIAL PRIMARY KEY
    CurrencyCode VARCHAR NOT NULL
)

INSERT INTO meta.currency(CurrencyCode)
VALUES('DKK')

CREATE TABLE meta.asset(
    id SERIAL PRIMARY KEY
    AssertType VARCHAR NOT NULL
)

INSERT INTO meta.asset(AssertType)
VALUES('Stock')

CREATE TABLE meta.exchange(
    id SERIAL PRIMARY KEY
    ExchangeId VARCHAR NOT NULL
)

INSERT INTO meta.exchange(ExchangeId)
VALUES('CSE')

CREATE TABLE meta.summary(
    id SERIAL PRIMARY KEY
    SummaryType VARCHAR NOT NULL
)

INSERT INTO meta.summary(SummaryType)
VALUES('Instrument')

CREATE TABLE meta.issuer(
    id SERIAL PRIMARY KEY,
    IssuerCountry VARCHAR
)

INSERT INTO meta.issuer(IssuerCountry)
VALUES('DK')

CREATE TABLE category(
    id SERIAL PRIMARY KEY,
    Category VARCHAR
)

INSERT INTO meta.issuer
VALUES('UNKNOWN')


