class matrix:

    def __init__(self,w_vertex,w_edge):
        self.Similarity_matrix_edge = {'method': {}, 'implements': {}, 'contains': {}, 'extends': {},'interface': {}}
        self.Similarity_matrix_edge['method']['method'] = w_vertex[0]
        self.Similarity_matrix_edge['extends']['extends'] = w_vertex[1]
        self.Similarity_matrix_edge['implements']['implements'] = w_vertex[2]
        self.Similarity_matrix_edge['contains']['contains'] = w_vertex[3]
        # self.Similarity_matrix_edge['interface']['interface'] = w_vertex[4]


        self.Similarity_matrix_edge['method']['implements'] = w_vertex[5]
        self.Similarity_matrix_edge['method']['contains'] = w_vertex[6]
        self.Similarity_matrix_edge['method']['extends'] = w_vertex[7]
        # self.Similarity_matrix_edge['method']['interface'] = w_vertex[8]

        self.Similarity_matrix_edge['implements']['method'] = w_vertex[9]
        self.Similarity_matrix_edge['implements']['contains'] = w_vertex[10]
        self.Similarity_matrix_edge['implements']['extends'] = w_vertex[11]
        # self.Similarity_matrix_edge['implements']['interface'] = w_vertex[12]

        self.Similarity_matrix_edge['contains']['method'] = w_vertex[13]
        self.Similarity_matrix_edge['contains']['implements'] = w_vertex[14]
        self.Similarity_matrix_edge['contains']['extends'] = w_vertex[15]
        # self.Similarity_matrix_edge['contains']['interface'] = w_vertex[16]


        self.Similarity_matrix_edge['extends']['method'] = w_vertex[17]
        self.Similarity_matrix_edge['extends']['implements'] = w_vertex[18]
        self.Similarity_matrix_edge['extends']['contains'] = w_vertex[19]
        # self.Similarity_matrix_edge['extends']['interface'] = w_vertex[20]

        self.Similarity_matrix_class = {'project': {}, 'method': {}, 'class': {}, 'interface':{}}
        self.Similarity_matrix_class['project']['project'] = w_edge[0]
        self.Similarity_matrix_class['method']['method'] = w_edge[1]
        self.Similarity_matrix_class['class']['class'] = w_edge[2]
        self.Similarity_matrix_class['interface']['interface'] = w_edge[3]

        self.Similarity_matrix_class['project']['method'] = w_edge[4]
        self.Similarity_matrix_class['project']['class'] = w_edge[5]
        self.Similarity_matrix_class['project']['interface'] = w_edge[6]

        self.Similarity_matrix_class['method']['project'] = w_edge[7]
        self.Similarity_matrix_class['method']['class'] = w_edge[8]
        self.Similarity_matrix_class['method']['interface'] = w_edge[9]

        self.Similarity_matrix_class['interface']['class'] = w_edge[10]
        self.Similarity_matrix_class['interface']['method'] = w_edge[11]
        self.Similarity_matrix_class['interface']['project'] = w_edge[12]

        self.Similarity_matrix_class['class']['project'] = w_edge[13]
        self.Similarity_matrix_class['class']['method'] = w_edge[14]
        self.Similarity_matrix_class['class']['interface'] = w_edge[15]

    def edge_matrix(self):
        return self.Similarity_matrix_edge

    def class_matrix(self):
        return self.Similarity_matrix_class



# class matrix:
#
#     def __init__(self):
#         self.Similarity_matrix_edge = {}
#         self.Similarity_matrix_edge['achieved by'] = {}
#         self.Similarity_matrix_edge['achieved by']['achieved by'] = 1
#         self.Similarity_matrix_edge['achieved by']['consists of'] = 0.5
#         self.Similarity_matrix_edge['achieved by'][''] = 0.25
#         self.Similarity_matrix_edge['achieved by']['+'] = 0.25
#         self.Similarity_matrix_edge['achieved by']['++'] = 0.25
#         self.Similarity_matrix_edge['achieved by']['--'] = 0.25
#         self.Similarity_matrix_edge['achieved by']['-'] = 0.25
#
#         self.Similarity_matrix_edge['consists of'] = {}
#         self.Similarity_matrix_edge['consists of']['consists of'] = 1
#         self.Similarity_matrix_edge['consists of'][''] = 0.25
#         self.Similarity_matrix_edge['consists of']['+'] = 0.25
#         self.Similarity_matrix_edge['consists of']['++'] = 0.25
#         self.Similarity_matrix_edge['consists of']['-'] = 0.25
#         self.Similarity_matrix_edge['consists of']['--'] = 0.25
#
#         self.Similarity_matrix_edge[''] = {}
#         self.Similarity_matrix_edge[''][''] = 1
#         self.Similarity_matrix_edge['']['+'] = 0.25
#         self.Similarity_matrix_edge['']['++'] = 0.25
#         self.Similarity_matrix_edge['']['-'] = 0.25
#         self.Similarity_matrix_edge['']['--'] = 0.25
#
#         self.Similarity_matrix_edge['+'] = {}
#         self.Similarity_matrix_edge['+']['+'] = 1
#         self.Similarity_matrix_edge['+']['++'] = 0.8
#         self.Similarity_matrix_edge['+']['-'] = 0
#         self.Similarity_matrix_edge['+']['--'] = 0
#
#         self.Similarity_matrix_edge['++'] = {}
#         self.Similarity_matrix_edge['++']['++'] = 1
#         self.Similarity_matrix_edge['++']['-'] = 0
#         self.Similarity_matrix_edge['++']['--'] = 0
#
#         self.Similarity_matrix_edge['-'] = {}
#         self.Similarity_matrix_edge['-']['-'] = 1
#         self.Similarity_matrix_edge['-']['--'] = 0.8
#
#         self.Similarity_matrix_edge['--'] = {}
#         self.Similarity_matrix_edge['--']['--'] = 1
#
#         self.Similarity_matrix_class = {'Task': {}, 'Quality': {}}
#
#         self.Similarity_matrix_class['Quality']['Quality'] = 1
#         self.Similarity_matrix_class['Task']['Task'] = 1
#         self.Similarity_matrix_class['Task']['Quality'] = 0.5
#         self.Similarity_matrix_class['Quality']['Task'] = 0.5
#
#
#     def edge_matrix(self):
#         return self.Similarity_matrix_edge
#
#     def class_matrix(self):
#         return self.Similarity_matrix_class
