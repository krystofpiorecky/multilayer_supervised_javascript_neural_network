sigmoid = x => 1 / (1 + Math.exp(-x))
sigmoid_derivative = x => x * (1 - x)

class Test
{
	constructor(_inputs, _expected_output)
	{
		this.inputs = _inputs;
		this.inputs.push(1);
		this.expected_output = _expected_output;

		return this;
	}
}

class NeuralNetwork
{
	constructor(_input_count, _hidden_count)
	{
		_input_count++;

		this.hidden_neurons = [];
		for(let i = 0; i < _hidden_count; i++)
		{
			this.hidden_neurons.push(
				new Neuron(_input_count)
			);
		}

		_hidden_count++;

		this.hidden_neurons.push(
			new BiasNeuron(_input_count)
		);


		this.output_neuron = new Neuron(_hidden_count);

		return this;
	}

	guess(_test)
	{
		let inputs = _test.inputs;
		let expected_output = _test.expected_output;
		let hidden_values = [];

		this.hidden_neurons.forEach(
			(neuron, index) =>
			{
				hidden_values.push(neuron.value(inputs));
			}
		);

		return this.output_neuron.value(hidden_values);
	}

	trainOnTests(_tests, _count)
	{
		for(let i = 0; i < _count; i++)
		{
			_tests.forEach(
				(test, index) =>
				{
					this.trainOnTest(test);
				}
			);
		}
	}

	trainOnTest(_test)
	{
		let inputs = _test.inputs;
		let expected_output = _test.expected_output;
		let hidden_values = [];

		this.hidden_neurons.forEach(
			(neuron, index) =>
			{
				hidden_values.push(neuron.value(inputs));
			}
		);

		let guess = this.output_neuron.value(hidden_values);
		let error = expected_output - guess;
		let adjustment = error * sigmoid_derivative(guess);

		this.output_neuron.adjust(adjustment, guess, hidden_values);

		this.hidden_neurons.forEach(
			(neuron, index) =>
			{
				neuron.adjust(adjustment, guess, inputs);
			}
		);
	}
}

class Neuron
{
	constructor(_input_count)
	{
		this.synapses = [];

		for(let i = 0; i < _input_count; i++)
		{
			this.synapses.push(
				new Synapse()
			);
		}

		return this;
	}

	value(_inputs)
	{
		let value = 0;

		_inputs.forEach(
			(input, index) =>
			{
				value += input * this.synapses[index].weight;
			}
		);

		value = sigmoid(value);

		return value;
	}

	adjust(_adjustment, _guess, _inputs)
	{
		this.synapses.forEach(
			(synapse, index) =>
			{
				synapse.adjust(_adjustment, _guess, _inputs[index]);
			}
		);
	}
}

class BiasNeuron extends Neuron
{
	value(_inputs)
	{
		return 1;
	}
}

class Synapse
{
	constructor(_input_index)
	{
		this.weight = 2 * Math.random() - 1;

		return this;
	}

	adjust(_adjustment, _guess, _input)
	{
		let adjustment = _input * _adjustment;
		
		this.weight += adjustment;
	}
}

let tests = [
	new Test([0, 0], 1),
	new Test([1, 1], 1),
	new Test([1, 0], 0),
	new Test([0, 1], 0)
];

let neural_network = new NeuralNetwork(2, 2);

neural_network.trainOnTests(tests, 100000);

neural_network.hidden_neurons.forEach(
	(item, index) => 
	{
		console.log(item);
	}
);

console.log(neural_network.output_neuron);

tests.forEach(
	(test, index) => 
	{
		console.log(neural_network.guess(test));
	}
);