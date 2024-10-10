<script>
	import { dataStore } from '../../stores/dataStore';

	let personal = $dataStore.PERSONAL;
	let contact = $dataStore.CONTACT;
	let meta = $dataStore.META;
</script>

<footer id="footer" class="bg-light-gray">
	<div class="container mx-auto font-sans py-12">
		<div class="flex justify-between items-center py-4">
			<div class="text-left">
				<p>
					&copy {meta.creation_year} /
					<a href={contact.github} class="text-gray-600 no-underline">
						{personal.name}
					</a>
				</p>
			</div>
			<div class="text-right">
				<a class="text-gray-600 no-underline" href={meta.src}>View Source</a>
			</div>
		</div>
	</div>
</footer>
