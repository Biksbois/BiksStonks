select distinct on (
	model_id, 
	used_companies, 
	metadata ->> 'columns'
) 
	time_unit,
	data_from,
	data_to,
	r_squared, 
	model_id,
	-- metadata ->> 'duration' as duration,
	metadata ->> 'columns' as columns,	
    -- metadata ->> 'Epoch' as epoch,
	-- metadata ->> 'window_size' as windows_size,
	-- metadata ->> 'Output_size' as output_size,
	used_companies
from score
where
	not time_unit in ('12H', '1D') and
	model_id = 3 AND (
        (
            'Sydbank A/S' =  ANY(used_companies) and 
            cardinality(used_companies) = 1
        ) OR (
            'Sydbank A/S' =  ANY(used_companies) and 
            'Danske Bank A/S' =  ANY(used_companies) and 
            'Jyske Bank A/S' =  ANY(used_companies) and 
            cardinality(used_companies) = 3
        )
    )AND
	data_to in ('2021-04-01T00:00:00', '2018-04-01T00:00:00') AND
	cardinality(columns) < 5
order by 
	model_id, 
	used_companies, 
	metadata ->> 'columns', 
	r_squared desc