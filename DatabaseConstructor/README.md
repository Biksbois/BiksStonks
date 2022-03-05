# How to set up Postgresql database

This folder contains the script to setup the database. This database will contain a databasse called `stonksdb` alongside different tables.

```mermaid
classDiagram
    Animal <|-- Duck
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class Duck{
        +String beakColor
        +swim()
        +quack()
    }
    class Fish{
        -int sizeInFeet
        -canEat()
    }
    class Zebra{
        +bool is_wild
        +run()
    }

```

1. Install Postgres [here](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads) and set it up.
2. 