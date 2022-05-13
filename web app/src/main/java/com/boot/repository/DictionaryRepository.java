package com.boot.repository;

import javax.transaction.Transactional;

import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.JpaSpecificationExecutor;
import org.springframework.stereotype.Repository;

import com.boot.entity.Dictionary;

@Transactional
@Repository
public interface DictionaryRepository extends JpaRepository<Dictionary, Integer>,JpaSpecificationExecutor<Dictionary>{

}
