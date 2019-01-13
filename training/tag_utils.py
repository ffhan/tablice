import re


def parse_xml_tags(xml):
    data = dict()

    matches = re.findall('<.*>.*</.*>', xml)

    if len(matches) == 1 and xml == matches[0]:
        match = matches[0]
        tag = ''
        content = ''

        start = False
        end = False
        for s in match.split('</')[0]:
            if s == '<':
                start = True
            elif s == '>':
                end = True
            else:
                if start and not end:
                    tag += s
                elif start and end:
                    content += s
        data[tag] = content
        return data

    for match in matches:
        data.update(parse_xml_tags(match))

    return data

test_input = '''<annotation>
	<folder>pozicije</folder>
	<filename>0052.png</filename>
	<path>/home/fhancic/Documents/Faks/5_semestar/Projekt/src/detektorTablica/tablice/pozicije/0052.png</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>1000</width>
		<height>1000</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>registration_plate</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<corners>
			<x1>533</x1>
			<y1>374</y1>
			<x2>534</x2>
			<y2>363</y2>
			<x3>581</x3>
			<y3>343</y3>
			<x4>582</x4>
			<y4>333</y4>
		</corners>
	</object>
</annotation>'''

if __name__ == '__main__':
    print(parse_xml_tags(test_input))
