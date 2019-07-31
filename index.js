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

	train(_tests, _count)
	{
		for(let i = 0; i < _count; i++)
		{
			let guesses = [];
			let hidden_neuron_values = [];
			let input_values = [];

			_tests.forEach(
				(test, index) =>
				{
					let inputs = test.inputs;
					let expected_output = test.expected_output;
					let hidden_values = [];

					this.hidden_neurons.forEach(
						(neuron, index) =>
						{
							hidden_values.push(neuron.value(inputs));
						}
					);

					let guess = this.output_neuron.value(hidden_values);

					guesses.push(guess);
					hidden_neuron_values.push(hidden_values);
					input_values.push(inputs);
				}
			);

			let adjustments = [];

			_tests.forEach(
				(test, index) =>
				{
					let expected_output = test.expected_output;
					let guess = guesses[index];
					let error = expected_output - guess;
					let adjustment = error * sigmoid_derivative(guess);

					adjustments.push(adjustment);
				}
			);

			this.output_neuron.adjust(adjustments, guesses, hidden_neuron_values);

			this.hidden_neurons.forEach(
				(neuron, index) =>
				{
					neuron.adjust(adjustments, guesses, input_values);
				}
			);
		}
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

	adjust(_adjustments, _guesses, _inputs)
	{
		this.synapses.forEach(
			(synapse, index) =>
			{
				synapse.adjust(_adjustments, _guesses, _inputs, index);
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

	adjust(_adjustments, _guesses, _inputs, _input_index)
	{
		let adjustment = 0;

		_adjustments.forEach(
			(test, index) =>
			{
				adjustment += _inputs[index][_input_index] * _adjustments[index];
			}
		);
		
		this.weight += adjustment;
	}
}

let tests = [
	new Test([0, 0], 1),
	new Test([1, 1], 0),
	new Test([1, 0], 1),
	new Test([0, 1], 0)
];

let neural_network = new NeuralNetwork(2, 2);

neural_network.train(tests, 100000);

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