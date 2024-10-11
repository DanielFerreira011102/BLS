<script lang="ts">
	export let index = 0;
	export let title = '';
	export let href = '';
</script>

<li>
	<a {href} class="text-lg text-black hover:text-gray-700" on:click>
		<span class="mr-6 font-mono text-gray-500">
			{(index + 1).toString().padStart(2, '0')}.
		</span>
		{title}
	</a>
</li>
