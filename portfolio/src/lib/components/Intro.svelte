<script lang="ts">
	import { Icon } from 'svelte-icons-pack';
	import { FaSolidArrowRight } from 'svelte-icons-pack/fa';
	import { dataStore } from '$lib/stores/dataStore';

	let personal = $dataStore.PERSONAL;
</script>

<section id="intro" class="container mx-auto py-12 font-sans">
	<h1 class="m-0 text-4xl font-semibold text-black">
		Hi, I'm {personal.first_name}
	</h1>
	<span class="text-2xl text-gray-500">
		{personal.role} from {personal.location}
	</span>
	<p class="text-dark-grayish-blue py-2 text-xl">{personal.introduction}</p>
	<div class="mt-4 font-mono font-bold">
		<a
			href="/about"
			class="inline-flex items-center bg-black px-4 py-2 text-white hover:bg-gray-900"
		>
			More Information
			<Icon src={FaSolidArrowRight} className="inline ml-2" />
		</a>
	</div>
</section>
