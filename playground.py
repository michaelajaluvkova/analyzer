from RevenueMaker import RevenueMakerClass
import pandas as pd


revenue = RevenueMakerClass(include_support_cost=False,
                            filter_by_year=False,
                            switch_product_group=False,
                            column1='',
                            column2='')

revenue.create_presentation()
revenue.one_big_graph()
