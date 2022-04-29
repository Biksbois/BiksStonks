select distinct on (
	model_id, 
	metadata ->> 'columns',
	data_from,
	time_unit
) 
	data_from,
	data_to,
	time_unit,
	round(r_squared, 4 ) as rsquared,
	round(mse, 4 ) as mse,
	round(mae, 4 ) as mae,
	model_id,
	metadata ->> 'columns' as columns,	
	metadata ->> 'WS' as windows_size,
	used_companies
from score
where
	not time_unit in ('12H', '1D') and
	model_id = 10 AND (
        (
            'Sydbank A/S' =  ANY(used_companies) and 
            cardinality(used_companies) = 1
        ) OR (
            'Sydbank A/S' =  ANY(used_companies) and 
            'Danske Bank A/S' =  ANY(used_companies) and 
            'Jyske Bank A/S' =  ANY(used_companies) and 
            cardinality(used_companies) = 3
        )
    ) AND
	data_to in ('2021-04-01T00:00:00', '2018-04-01T00:00:00') AND
	cardinality(columns) = 1
    
order by 
	model_id,
	CASE WHEN time_unit='5T' THEN 0
		 WHEN time_unit='15T' THEN 1
		 WHEN time_unit='30T' THEN 2
		 WHEN time_unit='45T' THEN 3
		 WHEN time_unit='1H' THEN 4
		 ELSE 5 END,
	data_from,
	metadata ->> 'columns',
	used_companies, 
	r_squared desc
