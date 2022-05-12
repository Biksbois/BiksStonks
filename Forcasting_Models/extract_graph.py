import argparse
import utils.DatasetAccess as db_access



def get_arg():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scoreid", type=int, required=True)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_arg()
    connection = db_access.get_connection()
    graph = db_access.get_graph_for_id(args.scoreid, connection)
    graph = graph.reset_index()
    graph.to_clipboard(sep=',')
    print("done")

    # select distinct on (model_id, used_companies, metadata ->> 'cols') id, r_squared, model_id, used_companies, metadata ->> 'cols' as columns 
    # from score 
    # order by model_id, used_companies, metadata ->> 'cols', r_squared desc




    

