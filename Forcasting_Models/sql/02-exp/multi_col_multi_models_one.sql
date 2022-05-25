select distinct on (
	model_id, 
	metadata ->> 'columns',
	data_from,
	data_to
	-- time_unit
) 
	time,
	data_from,
	data_to,
	time_unit,
	model_id,
	round(r_squared, 4 ) as rsquared,
	round(mse, 4 ) as mse,
	round(mae, 4 ) as mae,
	metadata ->> 'columns' as columns,	
	metadata ->> 'forecasted_points' as forecasted_points,
	metadata ->> 'windows_size' as windows_size,
	used_companies,
	metadata ->> 'train_epochs'
from score
where
	not time_unit in ('12H', '1D')  AND (
        (
            'Sydbank A/S' =  ANY(used_companies) and 
            'Danske Bank A/S' =  ANY(used_companies) and 
            'Jyske Bank A/S' =  ANY(used_companies) and 
            cardinality(used_companies) = 3
        )
    ) AND
	data_to in ('2021-04-01T00:00:00') AND
	cardinality(columns) > 1 AND
	time_unit = '5T' AND
	metadata ->> 'forecasted_points' = '1' AND
	-- metadata ->> 'columns' like '%"Markedsberetninger"%' AND
	metadata ->> 'columns' like '%"compound"%' AND
	use_sentiment = TRUE AND
	model_id = 10
order by 
	model_id, 
	metadata ->> 'columns',
	data_from,
	data_to,
	used_companies, 
	r_squared desc
