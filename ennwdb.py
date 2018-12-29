# coding=utf-8

"""
Операции с базой данных
"""

import sqlite3 as sql

class NeuralNetworkDB:
    def open(self, database_path):
        """
        Открыть существующую базу данных с информацией о нейронных сетях или создать новую.
        :param data_base_path:
        :return:
        """

        self.sqldb = sql.connect(database_path)
        self._create_structure()
        self._cursor = self.sqldb.cursor()

    def _create_structure(self):
        # query_create_config     = ''''''
        # query_create_inputs_all     = ''''''
        # query_create_outputs_all    = ''''''
        # query_create_nn_species_all = ''''''
        # query_create_synapses       = ''''''
        # query_create_neuron_bodies  = ''''''
        # query_create_nn_input       = ''''''
        # query_create_nn_output      = ''''''

        query_create_config = """
            CREATE TABLE IF NOT EXISTS config     (
                sid         TEXT UNIQUE,
                value       REAL
            )
        """

        query_create_inputs_all = """
            CREATE TABLE IF NOT EXISTS inputs_all (
                id          INTEGER PRIMARY KEY,
                sid         TEXT UNIQUE
            )
        """

        query_create_outputs_all = """
            CREATE TABLE IF NOT EXISTS outputs_all (
                id          INTEGER PRIMARY KEY,
                sid         TEXT UNIQUE
            )
        """

        query_create_nn_species_all = """
            CREATE TABLE IF NOT EXISTS nn_species_all (
                id                 INTEGER PRIMARY KEY,
                parent_id          INTEGER,
                hash               TEXT,
                is_alive           INTEGER,
                effective_deepness INTEGER,
                response_time      REAL,
                resolving_ability  REAL,
                quality            REAL,
                adaptability       REAL
            )
        """

        query_create_synapses = """
            CREATE TABLE IF NOT EXISTS synapses (
                id              INTEGER PRIMARY KEY,
                species_id      INTEGER,
                neuron_in_id    INTEGER,
                neuron_owner_id INTEGER,
                weight          REAL
            )
        """

        query_create_neuron_bodies = """
            CREATE TABLE IF NOT EXISTS neuron_bodies (
                id                          INTEGER PRIMARY KEY,
                species_id                  INTEGER,
                bias                        REAL,
                transfer_function_type      INTEGER,
                transfer_function_params    TEXT,
                axon_length                 INTEGER CHECK(axon_length > 0)
            )
        """

        query_create_nn_input = """
            CREATE TABLE IF NOT EXISTS nn_input (
                species_id                  INTEGER,
                neuron_id                   INTEGER,
                input_sid                   TEXT
            )
        """

        query_create_nn_output = """
            CREATE TABLE IF NOT EXISTS nn_output (
                species_id                  INTEGER,
                neuron_id                   INTEGER,
                output_sid                  TEXT
            )
        """

        c = self.sqldb.cursor()

        c.execute(query_create_config        )
        c.execute(query_create_inputs_all    )
        c.execute(query_create_outputs_all   )
        c.execute(query_create_nn_species_all)
        c.execute(query_create_synapses      )
        c.execute(query_create_neuron_bodies )
        c.execute(query_create_nn_input      )
        c.execute(query_create_nn_output     )

#        c.execute('CREATE INDEX IF NOT EXISTS idx_nn_species_all_001 ON nn_species_all (is_alive)   ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_synapses_001       ON synapses       (species_id) ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_neuron_bodies_001  ON neuron_bodies  (species_id) ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_nn_input_001       ON nn_input       (synapse_id) ')
#        c.execute('CREATE INDEX IF NOT EXISTS idx_nn_output_001      ON nn_output      (neuron_id)  ')

        c.execute("INSERT OR IGNORE INTO config (sid, value) VALUES ('alive_neural_network_queue_len'              , 100  )")
        c.execute("INSERT OR IGNORE INTO config (sid, value) VALUES ('mutator_neuron_deleting_probability_factor'  , 0.01 )")
        c.execute("INSERT OR IGNORE INTO config (sid, value) VALUES ('mutator_synapse_deleting_probability_factor' , 0.05 )")
        c.execute("INSERT OR IGNORE INTO config (sid, value) VALUES ('mutator_neuron_adding_probability_factor'    , 0.01 )")
        c.execute("INSERT OR IGNORE INTO config (sid, value) VALUES ('mutator_synapse_adding_probability_factor'   , 0.05 )")

        self.sqldb.commit()

    def load_config(self):
        q = self._cursor.execute("SELECT sid, value FROM config")
        return {o[0]: o[1] for o in q}

    def add_inputs(self, input_sid):
        self._cursor.execute("INSERT OR IGNORE INTO inputs_all (sid) VALUES ('" + input_sid + "')")
        return self._cursor.lastrowid

    def add_outputs(self, output_sid):
        self._cursor.execute("INSERT OR IGNORE INTO outputs_all (sid) VALUES ('" + output_sid + "')")
        return self._cursor.lastrowid

    def get_alive_species(self):
        """
        :return: список идентификаторов живых сетей
        """
        q = self._cursor.execute(
                "SELECT id FROM nn_species_all WHERE is_alive != 0")
        return [o[0] for o in q]

    def extinct_species(self, species_ids):
        """
        помечаем массив видов как вымершие
        """
        q = "UPDATE nn_species_all SET is_alive=0 WHERE id in (%s)" % ",".join([str(id) for id in species_ids])
        self._cursor.execute(q)

    def load_species(self, species_id):
        """
        :param species_id: идентификатор нейросети
        :return: кортеж:
                    * parent_id
                    * effective_deepness
                    * response_time
                    * resolving_ability
                    * quality
                    * adaptability
        """
        q = """
            SELECT
                  parent_id
                , hash
                , effective_deepness
                , response_time
                , resolving_ability
                , quality
                , adaptability
            FROM nn_species_all
            WHERE
               id = %d
        """ % species_id
        result = list(self._cursor.execute(q))
        assert len(result) == 1, "Requested species (id=%d) is not found" % species_id
        return result[0]

    def load_neurons_data(self, species_id):
        """
        Загрузка параметров нейронов из базы данных
        :param species_id: идентификатор нейросети
        :return: список кортежей (id, transfer_function_type, transfer_function_params, axon_length)
        """
        q = """
            SELECT
                  id
                , transfer_function_type
                , transfer_function_params
                , bias
                , axon_length
            FROM neuron_bodies
            WHERE
               species_id = %d
        """ % species_id
        return list(self._cursor.execute(q))

    def load_synapses_data(self, species_id):
        """
        Загрузка синапсов из базы данных
        :param species_id: идентификатор нейросети
        :return: список кортежей (weight, neuron_in_id, neuron_owner_id)
        """
        q = """
            SELECT
                  weight
                , neuron_in_id
                , neuron_owner_id
            FROM synapses
            WHERE
                species_id = %d
        """ % species_id
        return list(self._cursor.execute(q))

    def load_nn_inputs(self, species_id):
        """
        Загрузка информации о входных нейронах
        :param species_id: идентификатор нейросети
        :return: список кортежей (neuron_id, input_sid)
        """
        q = """
            SELECT
                  neuron_id
                , input_sid
            FROM nn_input
            WHERE
               species_id = %d
        """ % species_id
        return list(self._cursor.execute(q))

    def load_nn_outputs(self, species_id):
        """
        Загрузка информации о входных нейронах
        :param species_id: идентификатор нейросети
        :return: список кортежей (neuron_id, output_sid)
        """
        q = """
            SELECT
                  neuron_id
                , output_sid
            FROM nn_output
            WHERE
               species_id = %d
        """ % species_id
        return list(self._cursor.execute(q))

    def save_species(self, parent_id
                     , hash
                     , effective_deepness
                     , response_time
                     , resolving_ability
                     , quality
                     , adaptability):
        """
        Сохранение новой нейросети и таким образом получение для нее собственного идентификатора
        :param parent_id: идентификатор родительской нейросети
        :param effective_deepness: эффективная глубина нейросети
        :param response_time: время отклика (целое число, т.к. измеряется в тактах работы НС)
        :param resolving_ability: временнАя разрешающая способность нейросети
        :param quality: качество результата, выдаваемого сетью (без учета времени отклика)
        :param adaptability: единый кумулятивный параметр, определяющий насколько хороша данная НС.
                    Зависит от качества результата, времени отклика, количества нейронов и синапсов,...
        :return: собственный идентификатор нейросети
        """
        q = """
            INSERT INTO nn_species_all (
                  parent_id
                , hash
                , is_alive
                , effective_deepness
                , response_time
                , resolving_ability
                , quality
                , adaptability
            ) VALUES (
                %d, "%s", 1, %d, %.7f, %.7f, %.7f, %.7f
            )
        """ % (
            parent_id or 0
            , hash
            , effective_deepness
            , response_time
            , resolving_ability
            , quality or -1
            , adaptability or -1
        )
        self._cursor.execute(q)
        return self._cursor.lastrowid

    def save_neuron_body(self, species_id, transfer_function_type, transfer_function_params, bias, axon_length):
        """
        Сохранение данных в таблицу neuron_bodies
        :param species_id: идентификатор нейросети
        :param transfer_function_type: тип передаточной функции
        :param transfer_function_params: параметры передаточной функции (строка, содержащая массив чисел в формате json)
        :param axon_length: количество элеменов в аксоне нейрона
        :return: идентификатор нейрона
        """

        q = """
            INSERT INTO neuron_bodies(
                  species_id
                , transfer_function_type
                , transfer_function_params
                , bias
                , axon_length
            ) VALUES (
                %d, %d, '%s', %.7f, %d
            )
        """ % (
            species_id
            , transfer_function_type
            , transfer_function_params
            , bias
            , axon_length
        )
        self._cursor.execute(q)
        return self._cursor.lastrowid

    def save_synapse(self, species_id, nid_in, nid_own, weight):
        """
        Сохранение в БД информации о синапсе
        :param species_id: идентификатор нейросети
        :param nid_in: идентификатор входного нейрона
        :param nid_own: идентификатор  нейрона-владельца синапса
        :param weight: вес синапса
        :return: идентификатор синапса
        """
        q = """
        INSERT INTO synapses(
              species_id
            , neuron_in_id
            , neuron_owner_id
            , weight
        ) VALUES (
            %d, %d, %d, %.7f
        )
        """ % (
            species_id
            , nid_in
            , nid_own
            , weight
        )
        self._cursor.execute(q)
        return self._cursor.lastrowid

    def save_nn_inputs(self, species_id, neuron_id, input_sid):
        """
        Сохранение в БД информации о рецепторах нейросети
        :param species_id: идентификатор нейросети
        :param neuron_id: идентификатор нейрона-рецептора
        :param input_sid: текстовый идентификатор входных данных
        :return:
        """
        q = """
        INSERT INTO nn_input(
              species_id
            , neuron_id
            , input_sid
        ) VALUES (
            %d, %d, '%s'
        )
        """ % (
            species_id
            , neuron_id
            , input_sid
        )
        self._cursor.execute(q)

    def save_nn_outputs(self, species_id, neuron_id, output_sid):
        """
        Сохранение в БД информации о рецепторах нейросети
        :param species_id: идентификатор нейросети
        :param neuron_id: идентификатор нейрона-индикатора
        :param output_sid: текстовый идентификатор входных данных
        :return:
        """
        q = """
        INSERT INTO nn_output(
              species_id
            , neuron_id
            , output_sid
        ) VALUES (
            %d, %d, '%s'
        )
        """ % (
            species_id
            , neuron_id
            , output_sid
        )
        self._cursor.execute(q)
