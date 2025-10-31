in my opnion, the knowledge builder pipeline should be following:
0. of course, some configurations and initialization step is needed
1. load the 3GPP documents
2. named entities recognition with NLP/LLM...etc. but not use manual notation as possiple
    a. extract name of procedures as a lable of 'Procedure' entity
    b. for each procedure, extract name of network function (such as AMF, SMF, AUSF, etc) as a lable of 'NetworkFunction' entitty.
    c. for each procedure, extract name of message (such as Nausf_UEAuthentication_Authenticate Request) as a lable of 'Message' entitty.
    d. for each message, extract name of parameters (such as SUCI, SN name) or name of keys (such as 5G HE AV, RES*) that contains in Message as a lable of 'Parameter' or 'Key' entitty.
    e. there must be one or more step in each procedure, extract it and provide it a suitable lable for a 'Step' entity with name '{procedure name}_step_n'.
3.  relation extraction with NLP/LLM...etc. but not use manual notation as possiple
    I think there is no clear definitoin in documents for relations, so solution with NLP and/or LLM will be a suitable choice
4. here is the detailed step for extracting:
    a. split the document into sections.
    b. select sections contain with section.
    c. extract step in this procedure with the messages, parameters, keys.
    d. extract relations for this procedure between entities.
    e. extract entities releated to this procedure for the the section.
    f. construct knowledge graph for this procedure.
    g. move to next section conatained with section and back to step C.
    h. once all procedure in current document all extracted and constructed to knowledge graph, move to next document and back to step A.
    i. merge all knowledge graph, load it to database
5. with the extracted entities and relation, construct a knowledge graph and publish to neo4j database
6. once the graph is loaded into database, close the connection.
7. extract figure -> classify the figure (sequence diagram or not) -> identify the section is a prcedure or not (yes if sequence diagram) -> construct KG from sequence diagram (KG still need original text section title, parent title and description
 text)
-----requirment/hint--------
1. all NLP/LLM should run with GPU
2. the name of procedures will only be the section heading which section contains a figure show the figure shows the step flow and neutral language explain the detail for each step, such as "5G AKA".
    * if a section is a procedure, the heading must be the name of the procedure and contains a figure, but it does not mean all section contains figure is a procedure.
    * if a section is recognized as a procedure, the name of this procedure will be {title_of_father_section}_{section_title}
3. DO NOT consider some normal word/phrase as an entity, even it match the pattern (such as 3GPP, to request, one, PLMN, first, etc)
4. some example of relations between entities shows as follow:
    {NetworkFunction, INVOLVE, {procedure name}_step_n}
    {{procedure name}_step_n, FOLLOWED_BY, {procedure name}_step_n+1}
    {{procedure name}_step_n, CONTAINS, Parameter}
    {{procedure name}_step_n, SEND, Message}
    {Procedure, INVOKE, NetworkFunction}
    {Message, SEND_BY, NetworkFunction}
    {Message, SEND_TO, NetworkFunction}
    {{procedure name}_step_n, PART_OF, Procedure}
   the relations shows above is strictly required and the the format should be align with the example shows above, you can add other relation if you find anything worth noting.
5. split the complete code to multiple file for readability.
6. the final solution should be capable if document added, removed, or modified.
7. maybe utilize some better(suitable and modern) NLP/LLM as possiple, or invoke multiple NLP/LLM and select a specify module to each step.
8. every procedure needs to be searchable, i.e. i should get 5G AKA or EPA AKA procedure when search with "authentication process between UE and 5G core network".
9. for every step node, it should include description words from document.
10. the constructed knowledge graph should be able to convert into finite state machine for conformance test.
11. the document is docx file, 