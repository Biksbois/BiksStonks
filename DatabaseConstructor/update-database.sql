\c stonksdb

ALTER TYPE dataset_type
RENAME ATTRIBUTE category TO PrimaryCategory;

ALTER TYPE dataset_type
ADD ATTRIBUTE SecondaryCategory VARCHAR(100);

ALTER TABLE dataset
RENAME COLUMN Category TO PrimaryCategory;

ALTER TABLE dataset
ADD COLUMN SecondaryCategory VARCHAR(100);

ALTER TABLE dataset RENAME CONSTRAINT fk_category TO fk_primary_category;

ALTER TABLE dataset
ADD CONSTRAINT fk_secondary_category FOREIGN KEY(SecondaryCategory) REFERENCES metadata_category(category);